// Exact string-view inverted index. No approximate retrieval or hash-only
// equality: unordered containers compare full shingle bytes on every hit.
// candidates TSV: stable integer id, normalized body. stdin: source id, body.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using SV = std::string_view;
using Words = std::vector<std::pair<size_t,size_t>>;
Words words(SV s) {
    Words out;
    for (size_t a=0; a<s.size();) {
        auto b=s.find(' ',a); if (b==SV::npos) b=s.size();
        if (b>a) out.emplace_back(a,b);
        a=b+1;
    }
    return out;
}
SV gram(SV s,const Words& w,size_t i,size_t n) {
    return s.substr(w[i].first,w[i+n-1].second-w[i].first);
}
std::unordered_set<SV> unique_grams(SV s,const Words& w,size_t n) {
    std::unordered_set<SV> out;
    if(w.size()>=n) for(size_t i=0;i+n<=w.size();++i) out.insert(gram(s,w,i,n));
    return out;
}
struct Doc {
    std::string body;
    Words w;
    size_t grams5=0;
    std::vector<uint8_t> covered;
    bool exact=false,near=false;
    std::string first_source="-";
    double first_jaccard=0,first_containment=0;
};
int main(int argc,char** argv) {
  try {
    if(argc!=5) throw std::runtime_error("usage: match candidates.tsv output.tsv edges.tsv external|internal");
    bool internal=std::string(argv[4])=="internal";
    if(!internal && std::string(argv[4])!="external") throw std::runtime_error("invalid mode");
    std::ifstream cand(argv[1]); if(!cand) throw std::runtime_error("cannot open candidates");
    std::vector<Doc> docs;
    std::string line;
    while(std::getline(cand,line)) {
        auto t=line.find('\t'); if(t==std::string::npos) throw std::runtime_error("missing candidate tab");
        if(std::stoul(line.substr(0,t))!=docs.size()) throw std::runtime_error("nonsequential candidate id");
        Doc d; d.body=line.substr(t+1); docs.push_back(std::move(d));
    }
    if(docs.empty()) throw std::runtime_error("empty candidate input");
    // Build views only after vector reallocations stop (including SSO strings).
    std::unordered_map<SV,std::vector<int>> ix5,exact;
    std::unordered_map<SV,std::vector<std::pair<int,int>>> ix13;
    for(size_t d=0;d<docs.size();++d) {
        auto& v=docs[d]; v.w=words(v.body); v.covered.resize(v.w.size());
        auto gs=unique_grams(v.body,v.w,5); v.grams5=gs.size();
        for(auto g:gs) ix5[g].push_back(d);
        for(size_t i=0;i+13<=v.w.size();++i) ix13[gram(v.body,v.w,i,13)].emplace_back(d,i);
        exact[v.body].push_back(d);
    }
    // High document-frequency 13-grams are treated as shared templates only
    // for local coverage. They remain in exact/Jaccard/containment matching.
    size_t template_min=std::max<size_t>(100,(docs.size()+99)/100),templates=0;
    for(auto it=ix13.begin();it!=ix13.end();) {
        size_t df=0; int last=-1;
        for(auto [d,pos]:it->second) if(d!=last) {++df;last=d;}
        if(df>=template_min) {it=ix13.erase(it);++templates;} else ++it;
    }
    std::cerr<<"indexed "<<docs.size()<<" candidates, "<<ix5.size()<<" 5grams, "
             <<ix13.size()<<" 13grams, "<<templates<<" template 13grams\n";
    std::ofstream edges(argv[3]); if(!edges) throw std::runtime_error("cannot write edges");
    edges<<"candidate\tsource\treason\tjaccard\tcontainment\n";
    std::vector<int> counts(docs.size(),0),touched;
    size_t scanned=0;
    while(std::getline(std::cin,line)) {
        auto t=line.find('\t'); if(t==std::string::npos) throw std::runtime_error("missing reference tab");
        std::string source=line.substr(0,t);
        SV body(line.data()+t+1,line.size()-t-1);
        int self=internal?std::stoi(source):-1;
        auto w=words(body);
        auto exact_it=exact.find(body);
        if(exact_it!=exact.end()) for(int d:exact_it->second) {
            if(d==self || (internal && d>self)) continue;
            auto& v=docs[d];
            if(internal || !v.exact) edges<<d<<'\t'<<source<<"\texact\t1\t1\n";
            if(!v.exact && !v.near) {v.first_source=source;v.first_jaccard=v.first_containment=1;}
            v.exact=true;
        }
        auto gs=unique_grams(body,w,5);
        for(auto g:gs) {
            auto it=ix5.find(g); if(it==ix5.end()) continue;
            for(int d:it->second) {
                if(d==self || (internal && d>self) || (!internal && (docs[d].exact||docs[d].near))) continue;
                if(counts[d]++==0) touched.push_back(d);
            }
        }
        for(int d:touched) {
            auto& v=docs[d]; double common=counts[d];
            double jac=common/(v.grams5+gs.size()-common);
            double cont=common/std::min(v.grams5,gs.size());
            // Short reference fragments cannot veto an entire long article
            // through a tiny containment denominator; local reuse handles them.
            bool hit=jac>=0.8 || (cont>=0.8 && std::min(v.w.size(),w.size())>=50 && common>=20);
            if(hit) {
                if(internal || !v.near) edges<<d<<'\t'<<source<<"\tnear\t"<<jac<<'\t'<<cont<<'\n';
                if(!v.exact && !v.near) {v.first_source=source;v.first_jaccard=jac;v.first_containment=cont;}
                v.near=true;
            }
            counts[d]=0;
        }
        touched.clear();
        std::unordered_map<int,std::vector<uint8_t>> internal_covered;
        for(size_t i=0;i+13<=w.size();++i) {
            auto it=ix13.find(gram(body,w,i,13)); if(it==ix13.end()) continue;
            for(auto [d,pos]:it->second) {
                auto& v=docs[d];
                if(internal) {
                    if(d>=self) continue;
                    auto& c=internal_covered[d]; if(c.empty()) c.resize(v.w.size()+w.size());
                    std::fill(c.begin()+pos,c.begin()+pos+13,1);
                    std::fill(c.begin()+v.w.size()+i,c.begin()+v.w.size()+i+13,1);
                } else {
                    if(v.exact||v.near) continue;
                    std::fill(v.covered.begin()+pos,v.covered.begin()+pos+13,1);
                    if(v.first_source=="-") v.first_source=source;
                }
            }
        }
        for(const auto& [d,c]:internal_covered) {
            auto n=docs[d].w.size();
            auto a=std::count(c.begin(),c.begin()+n,1),b=std::count(c.begin()+n,c.end(),1);
            if(std::min(n,w.size())>=50 && (double(a)/n>=0.3 || double(b)/w.size()>=0.3))
                edges<<d<<'\t'<<source<<"\tpartial\t0\t0\n";
        }
        if(++scanned%100000==0) std::cerr<<"references "<<scanned<<'\n';
    }
    if(!std::cin.eof()) throw std::runtime_error("reference read failure");
    std::ofstream out(argv[2]); if(!out) throw std::runtime_error("cannot write summary");
    out<<"candidate\texact\tnear\tcovered_words\twords\tcoverage\tfirst_source\tjaccard\tcontainment\tcovered_bitmap\n";
    for(size_t d=0;d<docs.size();++d) {
        auto& v=docs[d]; size_t covered=std::count(v.covered.begin(),v.covered.end(),1);
        out<<d<<'\t'<<v.exact<<'\t'<<v.near<<'\t'<<covered<<'\t'<<v.w.size()<<'\t'
           <<(v.w.empty()?0:double(covered)/v.w.size())<<'\t'<<v.first_source<<'\t'
           <<v.first_jaccard<<'\t'<<v.first_containment<<'\t';
        for(auto bit:v.covered) out<<int(bit);
        out<<'\n';
    }
    if(!out || !edges) throw std::runtime_error("output write failure");
    std::cerr<<"complete references "<<scanned<<"\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
