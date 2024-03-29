# SChunk-Encoder
SChunk-Encoder (Transformer or Conformer) for streaming E2E ASR

SChunk-Encoder is implemented based on WeNet toolkit. And, it's motivated by Swin-Transformer in CV domain.
Particulary, what we have done in SChunk-Encoder is that——
1. We propose to use a shifted chunk mechanism instead of the conventional chunk mechanism for streaming Transformer and Conformer. This shifted chunk mechanism can significantly enhance modeling power through allowing chunk self-attention to
capture global context across local chunks, while keeping linear computational complexity and parallel trainable.
2. We name the Shifted Chunk Transformer and Conformer as SChunk-Transofromer and SChunk-Conformer, respectively. And we verify their performance on the widely used AISHELL-1 benckmark. Experiments show that the SChunk-Transformer and
SChunk-Conformer achieve CER 6.43% and 5.77%, respectively. That surpasses the existing chunk-wise and memory based methods by a large margin, and is competitive even compared with the state-of-the-art time-restricted methods which
have quadratic computational complexity.

More details can be found in our latest paper on arxiv:http://128.84.4.27/pdf/2203.15206 which has been accepted by ICONIP 2022.

We are avaiable through Email: fangyuan.wang@ia.ac.cn, feel free to contact us if any question you meet with our code.

Thanks WeNet team and Swin-Transformer team for their code sharing.

# Citation
If you find SChunk-Encoder useful in your work, you can cite the following paper:
```bibtex
@article{Wang2022ShiftedCE,
  title={Shifted Chunk Encoder for Transformer Based Streaming End-to-End ASR},
  author={Fangyuan Wang and Bo Xu},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.15206}
}
```
