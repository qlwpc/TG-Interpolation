// CPU-only fresh-segment metadata. No Torch/CUDA ABI or dense attention masks.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;
using Indices = std::vector<int>;
using Row = std::map<std::string, Indices>;

const std::vector<std::string> base_names = {
    "q_index", "k_index", "prefix", "sparse_index", "q_start", "q_count",
    "k_count", "sparse_count", "self_mask", "offsets", "edges", "reverse",
    "label_mask", "q_inverse", "k_inverse"};
const std::vector<std::string> tg_names = {
    "lo", "hi", "degree", "fwd_offsets", "fwd_keys", "bwd_offsets",
    "bwd_queries", "key_order"};

struct Parsed {
    Indices ordinary;
    Indices death;
};

// Parse once. Each key has at most two sparse consumers: its own compose/self
// row and the later compose row that pops it. Ordinary visibility is represented
// separately by prefixes (TGnomask/aug) or lifetime intervals (TG).
Parsed build_base(const long long *tokens, int n, int op, int oe, int cl, int ce,
                  int pad, bool augmented, Row &row) {
    for (const auto &name : base_names) row[name].assign(n, 0);
    auto &offsets = row["offsets"];
    auto &edges = row["edges"];
    auto &reverse = row["reverse"];
    auto &q_inverse = row["q_inverse"];
    auto &k_inverse = row["k_inverse"];
    offsets.assign(n + 1, 0);
    edges.clear();
    reverse.assign(2 * n, -1);
    q_inverse.assign(n, -1);
    k_inverse.assign(n, -1);
    row["label_mask"].assign(n, 1);

    Parsed parsed{{}, Indices(n, n)};
    auto &ordinary = parsed.ordinary;
    Indices stack, keys, sparse, prefixes, specials;
    long long last = -1;
    for (int i = 0; i < n; ++i) {
        const auto token = tokens[i];
        const bool closing = cl <= token && token < ce;
        bool compose = closing && last != token && last != -1;
        // A truncated segment may start in the middle of a closing-token run.
        if (closing && last == -1) {
            int end = i;
            while (end < n && tokens[end] == token) ++end;
            compose = (end - i) % 2 == 0;
        }
        Indices allowed;
        if (token == pad) {
            sparse.push_back(i);
            allowed.push_back(i);
        } else if (compose) {
            int j = i;
            while (!stack.empty() && !(op <= tokens[j] && tokens[j] < oe)) {
                j = stack.back();
                stack.pop_back();
                allowed.push_back(j);
                parsed.death[j] = i;
            }
            std::reverse(allowed.begin(), allowed.end());
            allowed.push_back(i);
            stack.push_back(i);
            keys.push_back(i);
            sparse.push_back(i);
            last = token;
        } else {
            if (!closing) {
                stack.push_back(i);
                keys.push_back(i);
            } else {
                row["label_mask"][i] = 0;
            }
            ordinary.push_back(i);
            prefixes.push_back(keys.size());
            specials.push_back(closing);
            last = token;
        }
        for (int key : allowed) {
            const int slot = 2 * key + (reverse[2 * key] != -1);
            if (reverse[slot] != -1)
                throw std::runtime_error("more than two compose consumers");
            reverse[slot] = i;
        }
        edges.insert(edges.end(), allowed.begin(), allowed.end());
        offsets[i + 1] = edges.size();
    }
    std::stable_sort(sparse.begin(), sparse.end(), [&](int i, int j) {
        return offsets[i + 1] - offsets[i] < offsets[j + 1] - offsets[j];
    });
    if (augmented) {
        // Augmented ordinary rows see all preceding positions, including pads
        // and repeated closes. Sparse compose/padding rows are unchanged.
        keys.resize(n);
        std::iota(keys.begin(), keys.end(), 0);
        for (int i = 0; i < int(ordinary.size()); ++i) {
            prefixes[i] = ordinary[i] + 1;
            specials[i] = 0;
        }
    }
    auto copy = [&](const std::string &name, const Indices &values) {
        std::copy(values.begin(), values.end(), row[name].begin());
    };
    copy("q_index", ordinary);
    copy("k_index", keys);
    copy("sparse_index", sparse);
    copy("prefix", prefixes);
    copy("self_mask", specials);
    edges.resize(2 * n, 0);
    row["q_count"] = {int(ordinary.size())};
    row["k_count"] = {int(keys.size())};
    row["sparse_count"] = {int(sparse.size())};
    for (int i = 0; i < int(ordinary.size()); ++i) q_inverse[ordinary[i]] = i;
    for (int i = 0; i < int(keys.size()); ++i) k_inverse[keys[i]] = i;
    for (int i = 0; i < n; ++i) {
        row["q_start"][i] = std::upper_bound(prefixes.begin(), prefixes.end(), i) - prefixes.begin();
    }
    return parsed;
}

int bit_length(int value) {
    int result = 0;
    while (value) {
        ++result;
        value >>= 1;
    }
    return result;
}

// Only TG needs lifetime/tile union schedules. TGnomask/aug retains O(B*N)
// storage; TG's union lists can be quadratic for degenerate trees.
void build_tg_schedule(const Parsed &parsed, int n, int q_tile, int k_tile, Row &row) {
    for (const auto &name : tg_names) row[name].assign(n, 0);
    const auto &ordinary = parsed.ordinary;
    auto &lo = row["lo"];
    auto &hi = row["hi"];
    auto &order = row["key_order"];
    Indices delta(ordinary.size() + 1, 0);
    for (int i = 0; i < n; ++i) {
        lo[i] = std::lower_bound(ordinary.begin(), ordinary.end(), i) - ordinary.begin();
        hi[i] = std::lower_bound(ordinary.begin(), ordinary.end(), parsed.death[i]) - ordinary.begin();
        if (row["k_inverse"][i] < 0) {
            lo[i] = std::max(row["q_inverse"][i], 0);
            hi[i] = std::max(row["q_inverse"][i] + 1, 0);
        }
        ++delta[lo[i]];
        --delta[hi[i]];
    }
    int count = 0;
    for (int i = 0; i < int(ordinary.size()); ++i) {
        count += delta[i];
        row["degree"][ordinary[i]] = count;
    }
    auto &fwd_offsets = row["fwd_offsets"];
    auto &fwd_keys = row["fwd_keys"];
    fwd_offsets = {0};
    fwd_keys.clear();
    for (int start = 0; start < n; start += q_tile) {
        const int end = std::min(start + q_tile, int(ordinary.size()));
        for (int k = 0; k < n; ++k) {
            if (lo[k] < end && hi[k] > start) fwd_keys.push_back(k);
        }
        fwd_offsets.push_back(fwd_keys.size());
    }
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        const int bi = bit_length(hi[i] - lo[i]), bj = bit_length(hi[j] - lo[j]);
        if (bi != bj) return bi < bj;
        if (lo[i] != lo[j]) return lo[i] < lo[j];
        return i < j;
    });
    auto &bwd_offsets = row["bwd_offsets"];
    auto &bwd_queries = row["bwd_queries"];
    bwd_offsets = {0};
    bwd_queries.clear();
    for (int start = 0; start < n; start += k_tile) {
        std::vector<std::pair<int, int>> intervals;
        for (int p = start; p < std::min(start + k_tile, n); ++p) {
            const int k = order[p];
            if (hi[k] > lo[k]) intervals.emplace_back(lo[k], hi[k]);
        }
        std::sort(intervals.begin(), intervals.end());
        int end = 0;
        for (const auto &interval : intervals) {
            for (int q = std::max(interval.first, end); q < interval.second; ++q) {
                bwd_queries.push_back(q);
            }
            end = std::max(end, interval.second);
        }
        bwd_offsets.push_back(bwd_queries.size());
    }
}

template <typename T>
py::array_t<T> export_array(const std::vector<Row> &rows, const std::string &name,
                           int width, const std::vector<py::ssize_t> &shape) {
    py::array_t<T> array(shape);
    auto *data = array.mutable_data();
    std::fill(data, data + rows.size() * width, T{});
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &values = rows[i].at(name);
        std::copy(values.begin(), values.end(), data + i * width);
    }
    return array;
}

py::dict build(py::array_t<long long, py::array::c_style | py::array::forcecast> ids,
               int op, int oe, int cl, int ce, int pad, int q_tile, int k_tile,
               bool include_tg, bool augmented) {
    if (ids.ndim() != 2 || ids.shape(0) == 0 || ids.shape(1) == 0)
        throw py::value_error("input_ids must be a nonempty 2-D array");
    if ((q_tile != 16 && q_tile != 32 && q_tile != 64) ||
        (k_tile != 16 && k_tile != 32 && k_tile != 64))
        throw py::value_error("TG tile sizes must be 16, 32 or 64");
    if (include_tg && augmented)
        throw py::value_error("TG lifetimes require non-augmented base metadata");
    const int b = ids.shape(0), n = ids.shape(1);
    int capacity = 0;
    std::vector<Row> rows(b);
    {
        const auto *tokens = ids.data();
        py::gil_scoped_release release;
        for (int batch = 0; batch < b; ++batch) {
            auto &row = rows[batch];
            const auto parsed = build_base(tokens + batch * n, n, op, oe, cl, ce, pad, augmented, row);
            capacity = std::max(capacity, row["sparse_count"][0]);
            if (include_tg) build_tg_schedule(parsed, n, q_tile, k_tile, row);
        }
    }
    auto names = base_names;
    if (include_tg) names.insert(names.end(), tg_names.begin(), tg_names.end());
    py::dict out;
    for (const auto &name : names) {
        const bool boolean = name == "self_mask" || name == "label_mask";
        const bool scalar = name == "q_count" || name == "k_count" || name == "sparse_count";
        int width = 1;
        for (const auto &row : rows) width = std::max(width, int(row.at(name).size()));
        std::vector<py::ssize_t> shape = {b};
        if (!scalar) shape.push_back(width);
        if (name == "reverse") shape = {b, n, 2};
        if (boolean) out[py::str(name)] = export_array<bool>(rows, name, width, shape);
        else out[py::str(name)] = export_array<int>(rows, name, width, shape);
    }
    out["sparse_capacity"] = capacity;
    return out;
}

PYBIND11_MODULE(_tg_layout_cpu, m) {
    m.def("build", &build, py::arg("ids"), py::arg("op"), py::arg("oe"),
          py::arg("cl"), py::arg("ce"), py::arg("pad"), py::arg("qt"), py::arg("kt"),
          py::arg("include_tg") = true, py::arg("augmented") = false);
}
