// Reuse only the lexical indexing primitives from the build-time matcher.
// This audit scans every shared shingle, retains maxima, and masks no templates.
#define main build_time_matcher_main
#include "../datatools/reserved_clean/match.cpp"
#undef main
#include <iomanip>

struct Best { double score=0; std::string source="-", body; };
struct Target {
    std::string body; Words w; size_t ng=0;
    bool exact=false; Best jac, cont, local;
    std::vector<unsigned char> covered;
};
void update(Best& best, double score, const std::string& source, SV body) {
    if(score>best.score) best={score,source,std::string(body)};
}
int main(int argc,char** argv) {
  try {
    if(argc!=3) throw std::runtime_error("usage: audit_reserved_fuzzy targets.tsv output.tsv");
    std::ifstream input(argv[1]); std::vector<Target> docs; std::string line;
    while(std::getline(input,line)) {
        auto t=line.find('\t');
        if(t==std::string::npos || std::stoul(line.substr(0,t))!=docs.size())
            throw std::runtime_error("invalid target index");
        Target d; d.body=line.substr(t+1); docs.push_back(std::move(d));
    }
    if(docs.empty()) throw std::runtime_error("empty targets");
    std::unordered_map<SV,std::vector<int>> ix5, exact;
    std::unordered_map<SV,std::vector<std::pair<int,int>>> ix13;
    for(size_t d=0;d<docs.size();++d) {
        auto& v=docs[d]; v.w=words(v.body); v.covered.resize(v.w.size());
        auto gs=unique_grams(v.body,v.w,5);v.ng=gs.size();
        for(auto g:gs) ix5[g].push_back(d);
        for(size_t i=0;i+13<=v.w.size();++i) ix13[gram(v.body,v.w,i,13)].emplace_back(d,i);
        exact[v.body].push_back(d);
    }
    std::vector<int> counts(docs.size()), touched;
    size_t scanned=0;
    while(std::getline(std::cin,line)) {
        auto t=line.find('\t');if(t==std::string::npos) throw std::runtime_error("invalid reference");
        std::string source=line.substr(0,t);SV body(line.data()+t+1,line.size()-t-1);
        auto w=words(body);auto gs=unique_grams(body,w,5);
        auto e=exact.find(body);if(e!=exact.end()) for(int d:e->second) docs[d].exact=true;
        for(auto g:gs) {
            auto it=ix5.find(g);if(it==ix5.end())continue;
            for(int d:it->second) if(counts[d]++==0)touched.push_back(d);
        }
        for(int d:touched) {
            auto& v=docs[d];double common=counts[d];
            update(v.jac,common/(v.ng+gs.size()-common),source,body);
            if(std::min(w.size(),v.w.size())>=50 && common>=20)
                update(v.cont,common/std::min(v.ng,gs.size()),source,body);
            counts[d]=0;
        }
        touched.clear();
        std::unordered_map<int,std::vector<int>> positions;
        for(size_t i=0;i+13<=w.size();++i) {
            auto it=ix13.find(gram(body,w,i,13));if(it==ix13.end())continue;
            for(auto [d,pos]:it->second) positions[d].push_back(pos);
        }
        for(auto& [d,pp]:positions) {
            auto& v=docs[d];std::sort(pp.begin(),pp.end());
            int end=-1, n=0;
            for(int p:pp) {
                n+=std::max(0,p+13-std::max(p,end));end=std::max(end,p+13);
                std::fill(v.covered.begin()+p,v.covered.begin()+p+13,1);
            }
            update(v.local,double(n)/v.w.size(),source,body);
        }
        if(++scanned%100000==0)std::cerr<<"references "<<scanned<<std::endl;
    }
    if(!std::cin.eof())throw std::runtime_error("reference read error");
    std::ofstream out(argv[2]);out<<std::setprecision(17);
    out<<"test_doc\texact\twords\tjaccard\tjac_source\tjac_body\tcontainment\tcont_source\tcont_body\tlocal_pair\tlocal_source\tlocal_body\tbitmap\n";
    for(size_t d=0;d<docs.size();++d) {
        auto& v=docs[d];out<<d<<'\t'<<v.exact<<'\t'<<v.w.size();
        for(const auto& b:{v.jac,v.cont,v.local})out<<'\t'<<b.score<<'\t'<<b.source<<'\t'<<b.body;
        out<<'\t';for(auto b:v.covered)out<<int(b);out<<'\n';
    }
    if(!out)throw std::runtime_error("output write error");
    std::cerr<<"complete references "<<scanned<<std::endl;
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
