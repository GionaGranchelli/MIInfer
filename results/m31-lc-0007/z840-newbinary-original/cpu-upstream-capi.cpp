#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  std::ifstream ids_file(argv[2]);
  std::vector<llama_token> ids;
  int id;
  while (ids_file >> id) ids.push_back(id);
  if (ids.empty()) return 3;
  llama_backend_init();
  auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
  auto *model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 4;
  auto cp = llama_context_default_params();
  cp.n_ctx = 2304; cp.n_batch = 512; cp.n_ubatch = 512;
  cp.n_threads = 48; cp.n_threads_batch = 48; cp.offload_kqv = false; cp.op_offload = false;
  auto *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 5;
  auto batch = llama_batch_init(512, 0, 1);
  std::ofstream out(argv[3], std::ios::binary);
  int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  std::vector<float> row(vocab);
  int generated[5];
  for (size_t pos=0; pos<ids.size();) {
    int n = std::min<size_t>(512, ids.size()-pos);
    for (int j=0;j<n;j++) { batch.token[j]=ids[pos+j]; batch.pos[j]=pos+j; batch.n_seq_id[j]=1; batch.seq_id[j][0]=0; batch.logits[j]=(pos+j==ids.size()-1); }
    batch.n_tokens=n;
    if (llama_decode(ctx,batch)!=0) return 6;
    pos += n;
  }
  for (int step=0; step<5; step++) {
    float *logits=llama_get_logits_ith(ctx,-1); if(!logits) return 7;
    std::copy(logits,logits+vocab,row.begin()); out.write(reinterpret_cast<char*>(row.data()),row.size()*sizeof(float));
    generated[step]=std::max_element(row.begin(),row.end())-row.begin();
    if(step<4) { batch.token[0]=generated[step]; batch.pos[0]=ids.size()+step; batch.n_seq_id[0]=1; batch.seq_id[0][0]=0; batch.logits[0]=1; batch.n_tokens=1; if(llama_decode(ctx,batch)!=0) return 8; }
  }
  out.close();
  for(int i=0;i<5;i++) std::cout<<(i?",":"")<<generated[i];
  std::cout<<" vocab="<<vocab<<"\n";
  llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
