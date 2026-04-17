#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <regex>
#include <limits>
#include <tuple>
#include <random>
#include <exception>

namespace py = pybind11;

std::pair<std::string, size_t> extract_object(const std::string& content, size_t start_pos, char start='{', char end='}') {
    int brace_count = 0;
    size_t obj_start = std::string::npos;
    
    for (size_t i = start_pos; i < content.size(); ++i) {
        if (content[i] == start) {
            brace_count++;
            if (obj_start == std::string::npos) obj_start = i;
        } else if (content[i] == end) {
            brace_count--;
            if (brace_count == 0) {
                return {content.substr(obj_start + 1, i - obj_start - 1), i};
            }
        }
    }
    return {"", std::string::npos};
}
/*
Auxiliary class for tokenizer to check tokens for terminals and non-terminals
*/
class SentencepieceVocab {
public:
    int64_t pad;
    int64_t bos;
    int64_t eos;
    int64_t unk;
    int64_t whitespace;
    int64_t newline;
    int64_t bosent;
    int64_t eosent;
    int64_t pause;
    std::pair<int64_t, int64_t> opening_non_terminals;
    std::pair<int64_t, int64_t> closing_non_terminals;
    std::vector<std::string> tokens;

    SentencepieceVocab() {}
    SentencepieceVocab(
        int64_t pad, int64_t bos, int64_t eos, int64_t unk,
        int64_t whitespace, int64_t newline, int64_t bosent, int64_t eosent, int64_t pause,
        std::pair<int64_t, int64_t> opening_non_terminals,
        std::pair<int64_t, int64_t> closing_non_terminals,
        std::vector<std::string> tokens
    ) : pad(pad), bos(bos), eos(eos), unk(unk),
        whitespace(whitespace), newline(newline), bosent(bosent), eosent(eosent), pause(pause),
        opening_non_terminals(opening_non_terminals),
        closing_non_terminals(closing_non_terminals),
        tokens(tokens) {}

    static SentencepieceVocab from_vocab_file(const std::string& vocab_file) {
        int64_t pad = -1, bos = -1, eos = -1, unk = -1;
        int64_t  whitespace = -1, newline = -1, bosent = -1, eosent = -1, pause = -1;
        std::pair<int64_t, int64_t> opening_non_terminals(-1, -1);
        std::pair<int64_t, int64_t> closing_non_terminals(-1, -1);
        std::vector<std::string> tokens;

        std::ifstream file(vocab_file);
        std::string line;
        int64_t index = 0;
        
        std::string json(".json");
        // vocab is hugging face style json tokenizer, 
        if (vocab_file.compare(vocab_file.size() - json.size(), json.size(), json) == 0)
        {
            try {
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();

                auto model_match = regex_search(content,  std::regex(R"("added_tokens"\s*:)") );
                if (!model_match) throw std::runtime_error("Missing model section");
                
                auto [added_content, added_end] = extract_object(content, content.find("added_tokens"), '[', ']');
                if (added_content.empty()) throw std::runtime_error("Invalid model object");

                // auto vocab_pos = model_content.find(R"("vocab")");
                // if (vocab_pos == std::string::npos) throw std::runtime_error("Missing vocab section");
                
                // auto [vocab_content, vocab_end] = extract_object(model_content, model_content.find('{', vocab_pos));
                // if (vocab_content.empty()) throw std::runtime_error("Invalid vocab object");
        
                //std::regex token_pattern(R"(\s*"((?:\\"|[^"])+)"\s*:\s*\d+)");
                // std::cout << "before" << '\n';
                
                std::regex token_pattern(R"("id"\s*:\s*(\d+)\s*,\s*"content"\s*:\s*"((?:\\"|[^"])+))");
                std::regex opening_pattern(R"(^<\([A-Z]+>$)");
                std::regex closing_pattern(R"(^<[A-Z]+\)>$)");
        
                // set<string> opening_non_terminals;
                // set<string> closing_non_terminals;
                // std::cout << added_content;
                std::sregex_iterator it(added_content.begin(), added_content.end(), token_pattern);
                std::sregex_iterator end;

        
                for (; it != end; ++it) {
                    std::smatch match = *it;
                    size_t index = std::stoi(match[1].str());
                    std::string token = regex_replace(match[2].str(), std::regex(R"(\\")"), "\"");
                    // std::cout<< "matched token " << token << '\n';
        
                    if (token == "<pad>" || token == "<|pad|>") pad = index;
                    else if (token == "<s>" || token=="<|beginoftext|>") bos = index;
                    else if (token == "</s>" || token=="<|endoftext|>") eos = index;
                    else if (token == "<unk>") unk = index;
                    else if (token == "(S1" || token == "(TOP") bosent = index;
                    else if (token == "S1)" || token == "TOP)") eosent = index;
                    else if (token == "▁" || token=="Ġ") whitespace = index;
                    else if (token == "<|SEP|>") pause = index;
                    else if (token == "Ċ") newline = index;
                    else if (std::regex_match(token, opening_pattern)) {
                        if (opening_non_terminals.first == -1)
                            opening_non_terminals.first = index;
                        opening_non_terminals.second = index+1;
                    }
                    else if (std::regex_match(token, closing_pattern)) {
                        if (closing_non_terminals.first == -1)
                            closing_non_terminals.first = index;
                        closing_non_terminals.second = index+1;
                    }
                }
        
            } catch (const std::exception& e) {
                std::cerr << "Error: in SentencepieceVocab " << e.what() << std::endl;
                exit(1);
            }

        }else // vocab is sentence piece vocabulary file
        {
            std::regex opening_pattern(R"(\([A-Z]+)");
            std::regex closing_pattern(R"([A-Z]+\))");

            while (std::getline(file, line)) {
                size_t tab_pos = line.find('\t');
                std::string token = line.substr(0, tab_pos);
                tokens.push_back(token);

                if (token == "<pad>") pad = index;
                else if (token == "<s>") bos = index;
                else if (token == "</s>") eos = index;
                else if (token == "<unk>") unk = index;
                else if (token == "(S1" || token == "(TOP") bosent = index;
                else if (token == "S1)" || token == "TOP)") eosent = index;
                else if (token == "▁") whitespace = index;
                else if (std::regex_match(token, opening_pattern)) {
                    if (opening_non_terminals.first == -1)
                        opening_non_terminals.first = index;
                    opening_non_terminals.second = index+1;
                }
                else if (std::regex_match(token, closing_pattern)) {
                    if (closing_non_terminals.first == -1)
                        closing_non_terminals.first = index;
                    closing_non_terminals.second = index+1;
                }

                index++;
            }
        }

        return SentencepieceVocab(
            pad, bos, eos, unk,
            whitespace, newline, bosent, eosent, pause,
            opening_non_terminals,
            closing_non_terminals,
            tokens
        );
    }

    bool is_opening_non_terminal(int64_t id) const {
        return id >= opening_non_terminals.first && 
               id < opening_non_terminals.second;
    }

    bool is_closing_non_terminal(int64_t id) const {
        return id >= closing_non_terminals.first && 
               id < closing_non_terminals.second;
    }

    bool is_non_terminal(int64_t id) const {
        return (id >= opening_non_terminals.first && 
                id < closing_non_terminals.second);
    }

    bool is_terminal(int64_t id) const {
        return id != pad && id != bos && id != eos && !is_non_terminal(id);
    }

    template <typename T>
    py::array_t<T> convert_treenpy_to_TG_impl(py::array_t<T> tree) {
        auto buf_tree = tree.template unchecked<1>();
        size_t N = tree.size();

        for (size_t i = 0; i < buf_tree.shape(0); ++i) 
            if (is_closing_non_terminal(buf_tree[i])) 
                N += 1;

        auto TG = py::array_t<T>(N);
        auto buf_TG = TG.template mutable_unchecked<1>();
        
        size_t tg_index = 0;
        for (size_t i = 0; i < buf_tree.shape(0); ++i) {
            T token = buf_tree[i];
            buf_TG[tg_index++] = token;
            if (is_closing_non_terminal(token))
                buf_TG[tg_index++] = token;
        }
        if (tg_index != N) {
            throw std::runtime_error("Assertion failed: tg_index != T");
        }
        return TG;
    }

    template <typename T>
    py::array_t<T> convert_treenpy_to_terminal_impl(py::array_t<T> tree) {
        auto buf_tree = tree.template unchecked<1>();
        size_t N = 0;

        for (size_t i = 0; i < buf_tree.shape(0); ++i) 
            if (!is_non_terminal(buf_tree[i])) 
                N += 1;

        auto term = py::array_t<T>(N);
        auto buf_term = term.template mutable_unchecked<1>();
        
        size_t tg_index = 0;
        for (size_t i = 0; i < buf_tree.shape(0); ++i) {
            T token = buf_tree[i];
            if (!is_non_terminal(token))
                buf_term[tg_index++] = token;
        }
        if (tg_index != N) {
            throw std::runtime_error("Assertion failed: terminal_index != T");
        }
        return term;
    }

    template <typename T>
    py::array_t<T> convert_TGnpy_to_tree_impl(py::array_t<T> TG_tree) {
        auto buf_TG = TG_tree.template unchecked<1>();
        size_t N = TG_tree.size();
        size_t tree_N = 0;
        for (size_t i = 0; i < buf_TG.shape(0); ++i) 
            if (is_closing_non_terminal(buf_TG[i]) && i < buf_TG.shape(0) - 1 && buf_TG[i+1]==buf_TG[i]) 
                tree_N++, i++;
            else
                tree_N++;

        auto tree = py::array_t<T>(tree_N);
        auto buf_tree = tree.template mutable_unchecked<1>();
        
        size_t tree_index = 0;
        for (size_t i = 0; i < buf_TG.shape(0); ++i) {
            T token = buf_TG[i];
            buf_tree[tree_index++] = token;
            if (is_closing_non_terminal(token) && i < buf_TG.shape(0) - 1 && buf_TG[i+1]==token)
                ++i;
        }
        if (tree_index != tree_N) {
            throw std::runtime_error("Assertion failed: tree_index != T");
        }
        return tree;
    }

    py::array convert_treenpy_to_TG(py::array tree) {
        auto dtype = tree.dtype();
        if (dtype.is(py::dtype::of<uint16_t>()))
            return convert_treenpy_to_TG_impl<uint16_t>(tree);
        else if (dtype.is(py::dtype::of<uint32_t>()))
            return convert_treenpy_to_TG_impl<uint32_t>(tree);
        else if (dtype.is(py::dtype::of<int64_t>()))
            return convert_treenpy_to_TG_impl<int64_t>(tree);
        else
            throw std::runtime_error("Unsupported data type");
    }

    py::array convert_treenpy_to_terminal(py::array tree) {
        auto dtype = tree.dtype();
        if (dtype.is(py::dtype::of<uint16_t>()))
            return convert_treenpy_to_terminal_impl<uint16_t>(tree);
        else if (dtype.is(py::dtype::of<uint32_t>()))
            return convert_treenpy_to_terminal_impl<uint32_t>(tree);
        else if (dtype.is(py::dtype::of<int64_t>()))
            return convert_treenpy_to_terminal_impl<int64_t>(tree);
        else
            throw std::runtime_error("Unsupported data type");
    }

    py::array convert_TGnpy_to_tree(py::array TG_tree) {
        auto dtype = TG_tree.dtype();
        if (dtype.is(py::dtype::of<uint16_t>()))
            return convert_TGnpy_to_tree_impl<uint16_t>(TG_tree);
        else if (dtype.is(py::dtype::of<uint32_t>()))
            return convert_TGnpy_to_tree_impl<uint32_t>(TG_tree);
        else if (dtype.is(py::dtype::of<int64_t>()))
            return convert_TGnpy_to_tree_impl<int64_t>(TG_tree);
        else
            throw std::runtime_error("Unsupported data type");
    }

    std::vector<int> select_from_range(int n, int k, std::mt19937 &gen) {
        std::vector<int> result;
        std::vector<int> range(n);
        std::iota(range.begin(), range.end(), 0);
        
        std::sample(range.begin(), range.end(), 
                    std::back_inserter(result),
                    k, gen);
        return result;
    }

    template <typename T>
    py::array_t<T> random_shuffle_tree_impl(py::array_t<T> TG_tree) {
        auto buf_TG = TG_tree.template unchecked<1>();
        size_t N = TG_tree.size();
        int64_t NT_k = 0;
        uint32_t seed = 0;
        std::uniform_int_distribution<int> dist(opening_non_terminals.first, closing_non_terminals.second - 1);
        for (size_t i = 0; i < buf_TG.shape(0); ++i) 
        {
            if (is_non_terminal(buf_TG[i])) 
                NT_k++;
            seed = seed*131 + buf_TG[i];
        }
        auto engine = std::mt19937{seed};
        auto random_positions = select_from_range(N, NT_k, engine);
        auto shuffle_tree = py::array_t<T>(N);
        auto buf_tree = shuffle_tree.template mutable_unchecked<1>();

        for (int i=0,j=0,p=0;i<N;++i)
        {
            if (j<NT_k && random_positions[j] == i) 
                buf_tree[i] = dist(engine), ++j;
            else
            {
                while(p<N && is_non_terminal(buf_TG[p])) ++p;
                buf_tree[i] = buf_TG[p], ++p;
            }
        }

        return shuffle_tree;
    }

    py::array random_shuffle_tree(py::array TG_tree) {
        auto dtype = TG_tree.dtype();
        if (dtype.is(py::dtype::of<uint16_t>()))
            return random_shuffle_tree_impl<uint16_t>(TG_tree);
        else if (dtype.is(py::dtype::of<uint32_t>()))
            return random_shuffle_tree_impl<uint32_t>(TG_tree);
        else if (dtype.is(py::dtype::of<int64_t>()))
            return random_shuffle_tree_impl<int64_t>(TG_tree);
        else
            throw std::runtime_error("Unsupported data type");
    }

    torch::Tensor get_non_terminal_mask(torch::Tensor input_ids) {
        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto label_acc = label_mask.accessor<bool, 1>();
        int64_t T = input_ids.size(0);
        for (int64_t i=0;i<T;++i) {
            int64_t token = input_acc[i];
            if (is_non_terminal(token))
                label_acc[i] = false;
        }
        return label_mask;
    }
};


// Helper class TG_Cache
class TG_Cache {
public:
    int64_t start;
    int64_t end;
    int64_t max_length;
    std::vector<int64_t> buffer; 

    TG_Cache () = default;
    TG_Cache(int64_t max_length, torch::Dtype dtype = torch::Dtype::Int) 
        : start(0), end(0), max_length(max_length),
          buffer(max_length, 0) {}

    void clear() {
        start = end = 0;
    }

    void add_end(int64_t T) {
        end = (end + T) % max_length;
    }

    int64_t& operator [] (int64_t i) {
        return buffer[(start + i) % max_length];
    }

    void append(const torch::Tensor& input_ids, bool update_state) {
        auto T = input_ids.size(0);
        auto input_data = input_ids.data_ptr<int64_t>();  // input_ids is Long Tensor
        if (end + T > max_length) {
            int64_t fir_len = max_length - end;
            std::memcpy(buffer.data() + end, input_data, fir_len * sizeof(int64_t));
            std::memcpy(buffer.data(), input_data + fir_len, (T - fir_len) * sizeof(int64_t));
            if (update_state) {
                end = T - fir_len;
            }
        } else {
            std::memcpy(buffer.data() + end, input_data, T * sizeof(int64_t));
            if (update_state) {
                end = (end + T) % max_length;
            }
        }
    }

    void pop_front(int64_t pop_length) {
        start = (start + pop_length) % max_length;
    }
    void pop_end(int64_t pop_length) {
        end = (end - pop_length + max_length) % max_length;
    }
};

// Main class TG_attention_bias
class TG_attention_bias {
public:
    SentencepieceVocab vocab_;
    std::vector<int64_t> stk_;
    std::vector<int64_t> stk_copy_;
    TG_Cache cached_input_;
    int64_t max_length_;
    
    // State variables
    int64_t last_token_;
    int64_t top_;
    int64_t cur_length_;

    TG_attention_bias() = default;
    TG_attention_bias(const std::string& vocab_path, int64_t max_token_length)
        : vocab_(SentencepieceVocab::from_vocab_file(vocab_path)),
          stk_(max_token_length * 2, 0), stk_copy_(max_token_length * 2, 0),
          cached_input_(max_token_length * 2, torch::kInt32),
          max_length_(max_token_length),
          last_token_(-1), top_(-1), cur_length_(0) {}

    void reset_state() {
        last_token_ = -1;
        top_ = -1;
        cur_length_ = 0;
        cached_input_.clear();
    }

    torch::Tensor convert_input_to_TG_format(const torch::Tensor& input_ids) {
        int64_t T = input_ids.size(0), len = T;
        auto input_acc = input_ids.accessor<int64_t, 1>();
        for (int64_t i = 0; i < T; ++i)
            len += vocab_.is_closing_non_terminal(input_acc[i]);
        auto TG_ids = torch::zeros({len}, torch::kLong);
        auto TG_acc = TG_ids.accessor<int64_t, 1>();
        for (int64_t i = 0, j = 0; i < T; ++i) {
            TG_acc[j++] = input_acc[i];
            if (vocab_.is_closing_non_terminal(input_acc[i]))
                TG_acc[j++] = input_acc[i];
        }
        return TG_ids;
    }

    bool should_compose(int64_t token, int64_t last_token, const torch::Tensor& input_ids, int64_t idx) {
        auto input_ptr = input_ids.data_ptr<int64_t>();
        if (vocab_.is_closing_non_terminal(token)) {
            if (last_token != token && last_token != -1) {
                return true;
            } else if (last_token == -1) {
                int64_t cnt = 0;
                while (input_ptr[idx] == token) {
                    cnt++;
                    idx++;
                    if (idx >= input_ids.size(0)) break;
                }
                return cnt % 2 == 0;
            }
        }
        return false;
    }

    std::tuple<torch::Tensor, torch::Tensor> operator()(const torch::Tensor& input_ids, bool update_state=false) {
        int64_t T = input_ids.size(0);
        int64_t update_T = T;
        int64_t top = top_;
        int64_t last_token = last_token_;
        cached_input_.append(input_ids, update_state);

        int64_t remove_len = (cur_length_ + T > max_length_) ? (cur_length_ + T - max_length_) : 0;
        int64_t pastT = (cur_length_ + T > max_length_) ? (max_length_ - T) : cur_length_;
        // Find stack beginning
        int64_t stk_beg = 0;
        bool found = false;
        for (int64_t i = 0; i <= top; ++i) {
            if (stk_[i] >= remove_len) {
                stk_beg = i;
                found = true;
                break;
            }
        }
        if (!found) stk_beg = top + 1;
        if (!update_state && top>=0) {
            std::memcpy(stk_copy_.data(), stk_.data(), (top+1) * sizeof(int64_t));
        }
        auto stk = update_state ? stk_.data() : stk_copy_.data();
        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto mask = torch::zeros({T, pastT + T}, torch::kBool);

        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto mask_acc = mask.accessor<bool, 2>();
        auto label_acc = label_mask.accessor<bool, 1>();
        for (int64_t i = 0; i < T; ++i) {
            int64_t token = input_acc[i];
            mask_acc[i][pastT + i] = true;
            // pad only occurs at the end of input, we won't cache pad into cached_input_
            if (token == vocab_.pad) {
                --update_T;
                continue;
            }
            if (should_compose(token, last_token, input_ids, i)) {
                int64_t j = cur_length_ + i;
                while (top >= stk_beg && !vocab_.is_opening_non_terminal(cached_input_[j])) {
                    j = stk[top];
                    top--;
                    mask_acc[i][j - remove_len] = true;
                }
                stk[++top] = cur_length_ + i;
            } else {
                if (!vocab_.is_closing_non_terminal(token) && token != vocab_.pad) {
                    stk[++top] = cur_length_ + i;
                } else {
                    label_acc[i] = false;
                }

                for (int64_t k = stk_beg; k <= top; ++k) {
                    mask_acc[i][stk[k] - remove_len] = true;
                }
            }

            last_token = token;
        }

        if (update_state) {
            // Update state variables
            top_ = top;
            last_token_ = last_token;
            cached_input_.pop_front(remove_len);
            cached_input_.pop_end(T - update_T);  //pop out pad tokens
            cur_length_ = cur_length_ - remove_len + update_T;

            // Update stack
            int64_t new_top = -1;
            for (int64_t i = stk_beg; i <= top; ++i) {
                int64_t val = stk_[i] - remove_len;
                if (val >= 0) {
                    stk_[++new_top] = val;
                }
            }
            top_ = new_top;
        }
        return std::make_tuple(mask, label_mask);
    }
};

class KProximal_TG_attention_bias : public TG_attention_bias {
public:
    int64_t prox_k_;
    TG_Cache cached_label_;
    bool is_aug_;

    KProximal_TG_attention_bias() = default;
    KProximal_TG_attention_bias(const std::string& vocab_path, int64_t max_token_length, int64_t Proximal_lenK, bool is_aug = false)
        : TG_attention_bias(vocab_path, max_token_length), 
          prox_k_(Proximal_lenK), cached_label_(max_token_length*2), is_aug_(is_aug) {}

    void reset_state() {
        TG_attention_bias::reset_state();
        cached_label_.clear();
    }

    std::tuple<torch::Tensor, torch::Tensor> operator()(const torch::Tensor& input_ids, bool update_state=false) {
        int64_t T = input_ids.size(0);
        int64_t update_T = T;
        int64_t top = top_;
        int64_t last_token = last_token_;
        cached_input_.append(input_ids, update_state);
        auto input_ptr = input_ids.data_ptr<int64_t>();

        int64_t remove_len = (cur_length_ + T > max_length_) ? (cur_length_ + T - max_length_) : 0;
        int64_t pastT = (cur_length_ + T > max_length_) ? (max_length_ - T) : cur_length_;
        int64_t doc_begin = 0;

        // Find stack beginning
        int64_t stk_beg = 0;
        bool found = false;
        for (int64_t i = 0; i <= top; ++i) {
            if (stk_[i] >= remove_len) {
                stk_beg = i;
                found = true;
                break;
            }
        }
        if (!found) stk_beg = top + 1;
        if (!update_state && top>=0) {
            std::memcpy(stk_copy_.data(), stk_.data(), (top+1) * sizeof(int64_t));
        }
        auto stk = update_state ? stk_.data() : stk_copy_.data();

        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto mask = torch::zeros({T, pastT + T}, torch::kBool);

        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto mask_acc = mask.accessor<bool, 2>();
        auto label_acc = label_mask.accessor<bool, 1>();

        for (int64_t i = 0; i < T; ++i) {
            mask_acc[i][pastT + i] = true;
            int64_t token = input_acc[i];
            if (token == vocab_.pad) {
                --update_T;
                continue;
            }

            if (should_compose(token, last_token, input_ids, i)) {
                int64_t j = cur_length_ + i;
                while (top >= stk_beg && !vocab_.is_opening_non_terminal(cached_input_[j])) {
                    j = stk[top];
                    top--;
                    mask_acc[i][j - remove_len] = true;
                }
                stk[++top] = cur_length_ + i;
            } else {
                if (!vocab_.is_closing_non_terminal(token) && token != vocab_.pad) {
                    stk[++top] = cur_length_ + i;
                } else {
                    label_acc[i] = false;
                }
                if (is_aug_)
                    for (int64_t k = std::max(pastT + i - prox_k_, doc_begin); k < pastT + i; ++k) {
                        mask_acc[i][k] = true;
                    }
                else
                    for (int64_t k = std::max(pastT + i - prox_k_, doc_begin); k < pastT + i; ++k) {
                        mask_acc[i][k] = cached_label_[k - pastT + cur_length_];
                    }
                for (int64_t k = stk_beg; k <= top; ++k) {
                    mask_acc[i][stk[k] - remove_len] = true;
                }
            }
            
            cached_label_[cur_length_ + i] = label_acc[i];


            last_token = token;
        }

        if (update_state) {
            // Update state variables
            top_ = top;
            last_token_ = last_token;
            cached_input_.pop_front(remove_len);
            cached_input_.pop_end(T - update_T);  //pop out pad tokens
            cached_label_.pop_front(remove_len);
            cached_label_.add_end(update_T);
            cur_length_ = cur_length_ - remove_len + update_T;

            // Update stack
            int64_t new_top = -1;
            for (int64_t i = stk_beg; i <= top; ++i) {
                int64_t val = stk_[i] - remove_len;
                if (val >= 0) {
                    stk_[++new_top] = val;
                }
            }
            top_ = new_top;
        }

        return std::make_tuple(mask, label_mask);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_alibi_rel_pos(const torch::Tensor& input_ids, bool update_state=false) {
        int64_t T = input_ids.size(0);
        int64_t top = top_;
        int64_t last_token = last_token_;
        cached_input_.append(input_ids, update_state);
        auto input_ptr = input_ids.data_ptr<int64_t>();

        int64_t remove_len = (cur_length_ + T > max_length_) ? (cur_length_ + T - max_length_) : 0;
        int64_t pastT = (cur_length_ + T > max_length_) ? (max_length_ - T) : cur_length_;
        int64_t doc_begin = 0;

        // Find stack beginning
        int64_t stk_beg = 0;
        bool found = false;
        for (int64_t i = 0; i <= top; ++i) {
            if (stk_[i] >= remove_len) {
                stk_beg = i;
                found = true;
                break;
            }
        }
        if (!found) stk_beg = top + 1;

        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto mask = torch::full({T, pastT + T}, std::numeric_limits<float>::lowest(), torch::kFloat32);
        auto rel_pos = torch::zeros({T, pastT + T}, torch::kFloat32);

        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto label_acc = label_mask.accessor<bool, 1>();
        auto mask_acc = mask.accessor<float, 2>();
        auto rel_pos_acc = rel_pos.accessor<float, 2>();

        for (int64_t i = 0; i < T; ++i) {
            mask_acc[i][pastT + i] = 0.0;
            int64_t token = input_acc[i];

            if (should_compose(token, last_token, input_ids, i)) {
                int64_t j = cur_length_ + i;
                while (top >= stk_beg && !vocab_.is_opening_non_terminal(cached_input_[j])) {
                    j = stk_[top];
                    top--;
                    mask_acc[i][j - remove_len] = 0.0;
                }
                stk_[++top] = cur_length_ + i;
            } else {
                if (!vocab_.is_closing_non_terminal(token) && token != vocab_.pad) {
                    stk_[++top] = cur_length_ + i;
                } else {
                    label_acc[i] = false;
                }
                
                for (int64_t k = doc_begin; k < pastT + i; ++k) {
                    if (cached_label_[k - pastT + cur_length_])
                        mask_acc[i][k] = 0.0, rel_pos_acc[i][k] = k - (pastT + i);
                }
            }
            
            cached_label_[cur_length_ + i] = label_acc[i];


            last_token = token;
            if (token == vocab_.eos) {
                top = -1;
                stk_beg = 0;
                doc_begin = pastT + i + 1;
            }
        }

        if (update_state) {
            // Update state variables
            top_ = top;
            last_token_ = last_token;
            cached_input_.pop_front(remove_len);
            cached_label_.pop_front(remove_len);
            cached_label_.add_end(T);
            cur_length_ = std::min(max_length_, cur_length_ + T);

            // Update stack
            int64_t new_top = -1;
            for (int64_t i = stk_beg; i <= top; ++i) {
                int64_t val = stk_[i] - remove_len;
                if (val >= 0) {
                    stk_[++new_top] = val;
                }
            }
            top_ = new_top;
        }

        return std::make_tuple(mask, rel_pos, label_mask);
    }
};

class Height_TG_attention_bias : public TG_attention_bias {
public:
    int64_t Height_H;
    TG_Cache cached_height_; // leaf and (NT is 1, NT') sub-tree height equals to max(child's height) + 1
                             // NT') has height 0.
    int64_t Prox_K_;
    Height_TG_attention_bias() = default;
    Height_TG_attention_bias(const std::string& vocab_path, int64_t max_token_length, int64_t Height_H, int64_t Prox_K=0)
        : TG_attention_bias(vocab_path, max_token_length), 
           Height_H(Height_H), cached_height_(max_token_length*2), Prox_K_(Prox_K) {}

    void reset_state() {
        TG_attention_bias::reset_state();
        cached_height_.clear();
    }

    std::tuple<torch::Tensor, torch::Tensor> operator()(const torch::Tensor& input_ids, bool update_state=false) {
        int64_t T = input_ids.size(0);
        int64_t top = top_;
        int64_t last_token = last_token_;
        cached_input_.append(input_ids, update_state);
        auto input_ptr = input_ids.data_ptr<int64_t>();

        int64_t remove_len = (cur_length_ + T > max_length_) ? (cur_length_ + T - max_length_) : 0;
        int64_t pastT = (cur_length_ + T > max_length_) ? (max_length_ - T) : cur_length_;
        int64_t doc_begin = 0;

        // Find stack beginning
        int64_t stk_beg = 0;
        bool found = false;
        for (int64_t i = 0; i <= top; ++i) {
            if (stk_[i] >= remove_len) {
                stk_beg = i;
                found = true;
                break;
            }
        }
        if (!found) stk_beg = top + 1;

        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto mask = torch::zeros({T, pastT + T}, torch::kBool);

        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto mask_acc = mask.accessor<bool, 2>();
        auto label_acc = label_mask.accessor<bool, 1>();

        for (int64_t i = 0; i < T; ++i) {
            mask_acc[i][pastT + i] = true;
            int64_t token = input_acc[i];
            if (token == vocab_.pad) 
                continue;

            if (should_compose(token, last_token, input_ids, i)) {
                int64_t j = cur_length_ + i;
                int64_t max_height=0;
                while (top >= stk_beg && !vocab_.is_opening_non_terminal(cached_input_[j])) {
                    j = stk_[top];
                    top--;
                    mask_acc[i][j - remove_len] = true;
                    max_height = std::max(max_height, cached_height_[j]);
                }
                stk_[++top] = cur_length_ + i;
                cached_height_[cur_length_ + i] = max_height + 1;
            } else {
                if (!vocab_.is_closing_non_terminal(token) && token != vocab_.pad) {
                    stk_[++top] = cur_length_ + i;
                    cached_height_[cur_length_ + i] = 1;
                } else {
                    label_acc[i] = false;
                    cached_height_[cur_length_ + i] = 0;
                }
                
                for (int64_t k = doc_begin; k < pastT + i; ++k) {
                    mask_acc[i][k] = cached_height_[k - pastT + cur_length_] >= Height_H || 
                                     (cached_height_[k - pastT + cur_length_]>0 && pastT + i - k <= Prox_K_);
                }
                for (int64_t k = stk_beg; k <= top; ++k) {
                    mask_acc[i][stk_[k] - remove_len] = true;
                }
            }
            
            last_token = token;
            // if (token == vocab_.eos) {
            //     top = -1;
            //     stk_beg = 0;
            //     doc_begin = pastT + i + 1;
            // }
        }

        if (update_state) {
            // Update state variables
            top_ = top;
            last_token_ = last_token;
            cached_input_.pop_front(remove_len);
            cached_height_.pop_front(remove_len);
            cached_height_.add_end(T);
            cur_length_ = std::min(max_length_, cur_length_ + T);

            // Update stack
            int64_t new_top = -1;
            for (int64_t i = stk_beg; i <= top; ++i) {
                int64_t val = stk_[i] - remove_len;
                if (val >= 0) {
                    stk_[++new_top] = val;
                }
            }
            top_ = new_top;
        }

        return std::make_tuple(mask, label_mask);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_alibi_rel_pos(const torch::Tensor& input_ids, bool update_state=false) {
        int64_t T = input_ids.size(0);
        int64_t top = top_;
        int64_t last_token = last_token_;
        cached_input_.append(input_ids, update_state);
        auto input_ptr = input_ids.data_ptr<int64_t>();

        int64_t remove_len = (cur_length_ + T > max_length_) ? (cur_length_ + T - max_length_) : 0;
        int64_t pastT = (cur_length_ + T > max_length_) ? (max_length_ - T) : cur_length_;
        int64_t doc_begin = 0;

        // Find stack beginning
        int64_t stk_beg = 0;
        bool found = false;
        for (int64_t i = 0; i <= top; ++i) {
            if (stk_[i] >= remove_len) {
                stk_beg = i;
                found = true;
                break;
            }
        }
        if (!found) stk_beg = top + 1;

        auto label_mask = torch::ones_like(input_ids, torch::kBool);
        auto mask = torch::full({T, pastT + T}, std::numeric_limits<float>::lowest(), torch::kFloat32);
        auto rel_pos = torch::zeros({T, pastT + T}, torch::kFloat32);

        auto input_acc = input_ids.accessor<int64_t, 1>();
        auto label_acc = label_mask.accessor<bool, 1>();
        auto mask_acc = mask.accessor<float, 2>();
        auto rel_pos_acc = rel_pos.accessor<float, 2>();

        for (int64_t i = 0; i < T; ++i) {
            mask_acc[i][pastT + i] = true;
            int64_t token = input_acc[i];

            if (should_compose(token, last_token, input_ids, i)) {
                int64_t j = cur_length_ + i;
                int64_t max_height=0;
                while (top >= stk_beg && !vocab_.is_opening_non_terminal(cached_input_[j])) {
                    j = stk_[top];
                    top--;
                    mask_acc[i][j - remove_len] = true;
                    max_height = std::max(max_height, cached_height_[j]);
                }
                stk_[++top] = cur_length_ + i;
                cached_height_[cur_length_ + i] = max_height + 1;
            } else {
                if (!vocab_.is_closing_non_terminal(token) && token != vocab_.pad) {
                    stk_[++top] = cur_length_ + i;
                    cached_height_[cur_length_ + i] = 1;
                } else {
                    label_acc[i] = false;
                    cached_height_[cur_length_ + i] = 0;
                }
                
                for (int64_t k = doc_begin; k < pastT + i; ++k) {
                    if (cached_height_[k - pastT + cur_length_])
                        mask_acc[i][k] = 0.0, 
                        rel_pos_acc[i][k] = cached_height_[k - pastT + cur_length_];
                }
            }
            
            last_token = token;
            if (token == vocab_.eos) {
                top = -1;
                stk_beg = 0;
                doc_begin = pastT + i + 1;
            }
        }

        if (update_state) {
            // Update state variables
            top_ = top;
            last_token_ = last_token;
            cached_input_.pop_front(remove_len);
            cached_height_.pop_front(remove_len);
            cached_height_.add_end(T);
            cur_length_ = std::min(max_length_, cur_length_ + T);

            // Update stack
            int64_t new_top = -1;
            for (int64_t i = stk_beg; i <= top; ++i) {
                int64_t val = stk_[i] - remove_len;
                if (val >= 0) {
                    stk_[++new_top] = val;
                }
            }
            top_ = new_top;
        }

        return std::make_tuple(mask, rel_pos, label_mask);
    }
};
    

class ChangeHead_attention_bias{
public:
    TG_attention_bias A;
    KProximal_TG_attention_bias B;

    std::tuple<torch::Tensor, torch::Tensor> operator()(const torch::Tensor& input_ids, bool update_state=false) {
        auto mask1 = A(input_ids, update_state);
        auto mask2 = B(input_ids, update_state);
        auto headmask1 = std::get<0>(mask1).unsqueeze(0).expand({6,-1,-1});
        auto headmask2 = std::get<0>(mask2).unsqueeze(0).expand({6,-1,-1});
        auto mask = torch::cat({headmask1, headmask2}, 0);
        return std::make_tuple(mask, std::get<1>(mask1));
    }
};

PYBIND11_MODULE(tg_mask, m) {
    py::class_<TG_Cache>(m, "TG_Cache")
        .def(py::init<int64_t, torch::Dtype>())
        .def("clear", &TG_Cache::clear)
        .def("append", &TG_Cache::append)
        .def("pop_front", &TG_Cache::pop_front)
        .def(py::init<const TG_Cache&>())
        .def(py::pickle(
            [](const TG_Cache &obj) { // __getstate__
                return py::make_tuple(
                    obj.start,          // SentencepieceVocab
                    obj.end,            // std::vector<int64_t>
                    obj.max_length,   // 假设 TG_Cache 已支持序列化
                    obj.buffer  
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) 
                    throw std::runtime_error("Invalid state for TG_Cache!");
    
                TG_Cache new_obj;
                new_obj.start = t[0].cast<int64_t>();
                new_obj.end   = t[1].cast<int64_t>();          
                new_obj.max_length = t[2].cast<int64_t>();     
                new_obj.buffer = t[3].cast<std::vector<int64_t>>();
    
                return new_obj;
            }
        ))
        ;

    py::class_<TG_attention_bias>(m, "TG_attention_bias")
        .def(py::init<const std::string&, int64_t>(), py::arg("vocab_path"), py::arg("max_token_length"))
        .def(py::init<const TG_attention_bias&>())
        .def("reset_state", &TG_attention_bias::reset_state)
        .def("convert_input_to_TG_format", &TG_attention_bias::convert_input_to_TG_format,
            py::arg("input_ids"))
        .def("__call__", &TG_attention_bias::operator(),
            py::arg("input_ids"), py::arg("update_state") = false)
        .def(py::pickle(
            [](const TG_attention_bias &obj) { // __getstate__
                return py::make_tuple(
                    obj.vocab_,          // SentencepieceVocab
                    obj.stk_,            // std::vector<int64_t>
                    obj.stk_copy_,       // std::vector<int64_t>
                    obj.cached_input_,   // 假设 TG_Cache 已支持序列化
                    obj.max_length_,     // int64_t
                    obj.last_token_,     // int64_t
                    obj.top_,            // int64_t
                    obj.cur_length_      // int64_t
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 8) 
                    throw std::runtime_error("Invalid state for TG_attention_bias!");
    
                TG_attention_bias new_obj;
                new_obj.vocab_ = t[0].cast<SentencepieceVocab>();
                new_obj.stk_ = t[1].cast<std::vector<int64_t>>();
                new_obj.stk_copy_ = t[2].cast<std::vector<int64_t>>();
                new_obj.cached_input_ = t[3].cast<TG_Cache>();
                new_obj.max_length_ = t[4].cast<int64_t>();
                new_obj.last_token_ = t[5].cast<int64_t>();
                new_obj.top_ = t[6].cast<int64_t>();
                new_obj.cur_length_ = t[7].cast<int64_t>();
    
                return new_obj;
            }
        ))    
        ;

    py::class_<KProximal_TG_attention_bias, TG_attention_bias>(m, "KProximal_TG_attention_bias")
        .def(py::init<const std::string&, int64_t, int64_t, bool>(), py::arg("vocab_path"), py::arg("max_token_length"), py::arg("Proximal_lenK"), py::arg("is_aug") = false)
        .def(py::init<const KProximal_TG_attention_bias&>())
        .def("reset_state", &KProximal_TG_attention_bias::reset_state)
        .def("convert_input_to_TG_format", &KProximal_TG_attention_bias::convert_input_to_TG_format,
            py::arg("input_ids"))
        .def("__call__", &KProximal_TG_attention_bias::operator(), 
            py::arg("input_ids"), py::arg("update_state") = false)
        .def("get_alibi_rel_pos", &KProximal_TG_attention_bias::get_alibi_rel_pos,
            py::arg("input_ids"), py::arg("update_state") = false)
        .def(py::pickle(
            [](const KProximal_TG_attention_bias &obj) { // __getstate__
                return py::make_tuple(
                    obj.vocab_,          // SentencepieceVocab
                    obj.stk_,            // std::vector<int64_t>
                    obj.stk_copy_,       // std::vector<int64_t>
                    obj.cached_input_,   // 假设 TG_Cache 已支持序列化
                    obj.max_length_,     // int64_t
                    obj.last_token_,     // int64_t
                    obj.top_,            // int64_t
                    obj.cur_length_,      // int64_t
                    obj.prox_k_, 
                    obj.cached_label_,
                    obj.is_aug_
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 11) 
                    throw std::runtime_error("Invalid state for KProximal_TG_attention_bias!");

                KProximal_TG_attention_bias new_obj;
                new_obj.vocab_ = t[0].cast<SentencepieceVocab>();
                new_obj.stk_ = t[1].cast<std::vector<int64_t>>();
                new_obj.stk_copy_ = t[2].cast<std::vector<int64_t>>();
                new_obj.cached_input_ = t[3].cast<TG_Cache>();
                new_obj.max_length_ = t[4].cast<int64_t>();
                new_obj.last_token_ = t[5].cast<int64_t>();
                new_obj.top_ = t[6].cast<int64_t>();
                new_obj.cur_length_ = t[7].cast<int64_t>();
                new_obj.prox_k_     = t[8].cast<int64_t>();
                new_obj.cached_label_ = t[9].cast<TG_Cache>();
                new_obj.is_aug_     = t[10].cast<bool>();
    
                return new_obj;
            }
        ))    
        ;
    
    py::class_<Height_TG_attention_bias, TG_attention_bias>(m, "Height_TG_attention_bias")
        .def(py::init<const std::string&, int64_t, int64_t, int64_t>(), py::arg("vocab_path"), py::arg("max_token_length"), py::arg("Height_H"), py::arg("Prox_K") = 0)
        .def(py::init<const Height_TG_attention_bias&>())
        .def("reset_state", &Height_TG_attention_bias::reset_state)
        .def("convert_input_to_TG_format", &Height_TG_attention_bias::convert_input_to_TG_format,
            py::arg("input_ids"))
        .def("__call__", &Height_TG_attention_bias::operator(), 
            py::arg("input_ids"), py::arg("update_state") = false)
        .def("get_alibi_rel_pos", &Height_TG_attention_bias::get_alibi_rel_pos,
            py::arg("input_ids"), py::arg("update_state") = false)
        .def(py::pickle(
            [](const Height_TG_attention_bias &obj) { // __getstate__
                return py::make_tuple(
                    obj.vocab_,          // SentencepieceVocab
                    obj.stk_,            // std::vector<int64_t>
                    obj.cached_input_,   // 假设 TG_Cache 已支持序列化
                    obj.max_length_,     // int64_t
                    obj.last_token_,     // int64_t
                    obj.top_,            // int64_t
                    obj.cur_length_,     // int64_t
                    obj.Height_H, 
                    obj.cached_height_,
                    obj.Prox_K_
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 10) 
                    throw std::runtime_error("Invalid state for Height_TG_attention_bias!");

                Height_TG_attention_bias new_obj;
                new_obj.vocab_ = t[0].cast<SentencepieceVocab>();
                new_obj.stk_ = t[1].cast<std::vector<int64_t>>();
                new_obj.cached_input_ = t[2].cast<TG_Cache>();
                new_obj.max_length_ = t[3].cast<int64_t>();
                new_obj.last_token_ = t[4].cast<int64_t>();
                new_obj.top_ = t[5].cast<int64_t>();
                new_obj.cur_length_ = t[6].cast<int64_t>();
                new_obj.Height_H     = t[7].cast<int64_t>();
                new_obj.cached_height_ = t[8].cast<TG_Cache>();
                new_obj.Prox_K_ = t[9].cast<int64_t>();

                return new_obj;
            }
        ))    
        ;

    // py::class_<TG_attention_bias_nomask, TG_attention_bias>(m, "TG_attention_bias_nomask")
    // .def(py::init<const std::string&, int64_t>(), py::arg("vocab_path"), py::arg("max_token_length"))
    // .def("reset_state", &TG_attention_bias_nomask::reset_state)
    // .def("__call__", &TG_attention_bias_nomask::operator(), 
    //         py::arg("input_ids"), py::arg("update_state") = false);
        
    py::class_<SentencepieceVocab>(m, "SentencepieceVocab")
        .def(py::init<
            int64_t, int64_t, int64_t, int64_t, 
            int64_t, int64_t, int64_t, int64_t,
            std::pair<int64_t, int64_t>,
            std::pair<int64_t, int64_t>,
            std::vector<std::string>>())
        .def(py::init<const SentencepieceVocab&>())
        .def_static("from_vocab_file", &SentencepieceVocab::from_vocab_file)
        .def("is_opening_non_terminal", &SentencepieceVocab::is_opening_non_terminal)
        .def("is_closing_non_terminal", &SentencepieceVocab::is_closing_non_terminal)
        .def("is_non_terminal", &SentencepieceVocab::is_non_terminal)
        .def("is_terminal", &SentencepieceVocab::is_terminal)
        .def_readwrite("pad", &SentencepieceVocab::pad)
        .def_readwrite("bos", &SentencepieceVocab::bos)
        .def_readwrite("eos", &SentencepieceVocab::eos)
        .def_readwrite("unk", &SentencepieceVocab::unk)
        .def_readwrite("whitespace", &SentencepieceVocab::whitespace)
        .def_readwrite("newline", &SentencepieceVocab::newline)
        .def_readwrite("bosent", &SentencepieceVocab::bosent)
        .def_readwrite("eosent", &SentencepieceVocab::eosent)
        .def_readwrite("pause", &SentencepieceVocab::pause)
        .def_readwrite("opening_non_terminals", &SentencepieceVocab::opening_non_terminals)
        .def_readwrite("closing_non_terminals", &SentencepieceVocab::closing_non_terminals)
        .def("convert_treenpy_to_TG", &SentencepieceVocab::convert_treenpy_to_TG, 
            py::arg("tree"), "Convert tree sequence to TG format")
        .def("convert_treenpy_to_terminal", &SentencepieceVocab::convert_treenpy_to_terminal, 
            py::arg("tree"), "Convert tree/TG sequence to terminal format")
        .def("convert_TGnpy_to_tree", &SentencepieceVocab::convert_TGnpy_to_tree, 
            py::arg("TG_tree"), "Convert TG sequence to tree format")
        .def("random_shuffle_tree", &SentencepieceVocab::random_shuffle_tree,
                py::arg("TG_tree"), "Convert Tree sequence to Random_tree format")
        .def("get_non_terminal_mask", &SentencepieceVocab::get_non_terminal_mask, 
            py::arg("input_ids"), "Generate tree Sequence Non terminal mask")
        // Add other property readers...
        .def(py::pickle(
            [](const SentencepieceVocab &v) { // __getstate__
                return py::make_tuple(
                    v.pad,
                    v.bos,
                    v.eos,
                    v.unk,
                    v.whitespace,
                    v.newline,
                    v.bosent,
                    v.eosent,
                    v.pause,
                    v.opening_non_terminals,
                    v.closing_non_terminals,
                    v.tokens
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 12)
                    throw std::runtime_error("Invalid state!");
                    
                return SentencepieceVocab(
                    t[0].cast<int64_t>(),
                    t[1].cast<int64_t>(),
                    t[2].cast<int64_t>(),
                    t[3].cast<int64_t>(),
                    t[4].cast<int64_t>(),
                    t[5].cast<int64_t>(),
                    t[6].cast<int64_t>(),
                    t[7].cast<int64_t>(),
                    t[8].cast<int64_t>(),
                    t[9].cast<std::pair<int64_t, int64_t>>(),
                    t[10].cast<std::pair<int64_t, int64_t>>(),
                    t[11].cast<std::vector<std::string>>()
                );
            }
        ))
        ;
}
