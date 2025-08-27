[![codecov](https://codecov.io/gh/gnina/gnina/branch/master/graph/badge.svg?token=kOC9GUKEGa)](https://codecov.io/gh/gnina/gnina)
[![Github CI](https://github.com/gnina/gnina/actions/workflows/CI.yml/badge.svg)](https://github.com/gnina/gnina/actions)

gnina (pronounced NEE-na) is a molecular docking program with integrated support for scoring and optimizing ligands using convolutional neural networks. It is a fork of [smina](http://smina.sf.net/), which is a fork of [AutoDock Vina](http://vina.scripps.edu/).  

Help
====
 
Please [subscribe to our slack team](https://join.slack.com/t/gninacnn/shared_invite/enQtNTY3ODk2ODk5OTU5LTkzMjY1ZTE3YjJlZmIxOWI2OTU3Y2RlMTIyYmM2YmFmYTU1NTk5ZTBmMjUwMGRhYzk1ZjY5N2E4Y2I5YWU5YWI).
An example colab notebook showing how to use gnina is available <a href="https://colab.research.google.com/drive/1QYo5QLUE80N_G28PlpYs6OKGddhhd931?usp=sharing">here</a>. We also hosted a workshp on using gnina ([video](https://www.youtube.com/watch?v=MG3Srzi5kZ0), [slides](https://bits.csb.pitt.edu/rsc_workshop2021/docking_with_gnina.slides.html#/)).

Citation
========
If you find gnina useful, please cite our paper(s):  


**GNINA 1.3: the next increment in molecular docking with deep learning** (Primary application citation)  
A McNutt, Y Li, R Meli, R Aggarwal, DR Koes. *J. Cheminformatics*, 2025  
[link](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00973-x) [PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874439/)

**GNINA 1.0: Molecular docking with deep learning** (Primary application citation, previous version)  
A McNutt, P Francoeur, R Aggarwal, T Masuda, R Meli, M Ragoza, J Sunseri, DR Koes. *J. Cheminformatics*, 2021  
[link](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2) [PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8191141/) [ChemRxiv](https://chemrxiv.org/articles/preprint/GNINA_1_0_Molecular_Docking_with_Deep_Learning/13578140)

**Protein–Ligand Scoring with Convolutional Neural Networks**  (Primary methods citation)  
M Ragoza, J Hochuli, E Idrobo, J Sunseri, DR Koes. *J. Chem. Inf. Model*, 2017  
[link](http://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740) [PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5479431/) [arXiv](https://arxiv.org/abs/1612.02751)  

**Virtual Screening with Gnina 1.0** (Virtual screening citation)
J Sunseri, DR Koes D. *Molecules*, 2021
[link](https://www.mdpi.com/1420-3049/26/23/7369) [Preprints](https://www.preprints.org/manuscript/202111.0329/v1)

**CACHE Challenge# 1: Docking with GNINA Is All You Need** (Example application)
I Dunn, S Pirhadi, Y Wang, S Ravindran, C Concepcion, DR Koes. *J. Chem. Inf. Model*, 2024 
[link](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01429) [PubMed](https://pubmed.ncbi.nlm.nih.gov/39654129/)

**Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design** (Dataset citation)
PG Francoeur, T Masuda, J Sunseri, A Jia, RB Iovanisci, I Snyder, DR Koes. *J. Chem. Inf. Model*, 2020  
[link](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00411) [PubMed](https://pubmed.ncbi.nlm.nih.gov/32865404/) [Chemrxiv](https://chemrxiv.org/articles/preprint/3D_Convolutional_Neural_Networks_and_a_CrossDocked_Dataset_for_Structure-Based_Drug_Design/11833323/1)

**Ligand pose optimization with atomic grid-based convolutional neural networks**  
M Ragoza, L Turner, DR Koes. *Machine Learning for Molecules and Materials NIPS 2017 Workshop*, 2017  
[arXiv](https://arxiv.org/abs/1710.07400)  

**Visualizing convolutional neural network protein-ligand scoring**  
J Hochuli, A Helbling, T Skaist, M Ragoza, DR Koes.  *Journal of Molecular Graphics and Modelling*, 2018  
[link](https://www.sciencedirect.com/science/article/pii/S1093326318301670) [PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6343664/) [arXiv](https://arxiv.org/abs/1803.02398)

**Convolutional neural network scoring and minimization in the D3R 2017 community challenge**  
J Sunseri, JE King, PG Francoeur, DR Koes.  *Journal of computer-aided molecular design*, 2018  
[link](https://link.springer.com/article/10.1007/s10822-018-0133-y) [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/29992528)



Docker
======

A pre-built docker image is available [here](https://hub.docker.com/u/gnina) and Dockerfiles are [here](https://github.com/gnina/gnina/tree/master/docker).

Installation
============

We recommend that you use the [pre-built binary](https://github.com/gnina/gnina/releases/latest) unless you have significant experience building software on Linux, in which case building from source *might* result in an executable more optimized for your system.  The pre-built binary can be used in WSL.

### Ubuntu 22.04
```
apt-get  install build-essential git cmake wget libboost-all-dev libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libhdf5-dev libatlas-base-dev python3-dev librdkit-dev python3-numpy python3-pip python3-pytest libjsoncpp-dev

```

[Follow NVIDIA's instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4TWipdwX1) to install the latest version of CUDA (>= 12.0 is required). **Make sure `nvcc` is in your PATH.**


#
Install OpenBabel3.  Note there are errors in bond order determination in version 3.1.1 and older.
```
git clone https://github.com/dkoes/openbabel.git
cd openbabel
mkdir build
cd build
cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON ..
make
make install
```

#
Install gnina
```
git clone https://github.com/gnina/gnina.git
cd gnina
mkdir build
cd build
cmake ..  # -DUSE_SYSTEM_NVTX=1 may be needed with pytorch 2.7.0 and CUDA 12.9
make
make install
```

### [WSL2 Ubuntu 22.04](https://github.com/gnina/gnina/issues/247)
```bash
sudo apt-get remove nvidia-cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
chmod 700 cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
apt-get install build-essential git cmake wget libboost-all-dev libeigen3-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler libhdf5-dev libatlas-base-dev python3-dev librdkit-dev python3-numpy python3-pip python3-pytest libjsoncpp-dev

git clone https://github.com/openbabel/openbabel.git
cd openbabel
mkdir build
cd build
cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON ..
make -j8
sudo make install

git clone https://github.com/gnina/gnina.git
cd gnina
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

#
If you are building for systems with different GPUs (e.g. in a cluster environment), configure with `-DCMAKE_CUDA_ARCHITECTURES=all`.   
Note that the cmake build will automatically fetch and install [libmolgrid](https://github.com/gnina/libmolgrid) and [torch](https://github.com/pytorch/pytorch) if they are not already installed.


The scripts provided in `gnina/scripts` have additional python dependencies that must be installed. 

Usage
=====
🔍 한 줄씩 설명
## 1
gnina
단백질-리간드 도킹을 수행하는 프로그램 실행 명령어야.
AutoDock Vina 기반이지만, deep learning scoring 기능이 추가된 버전이야.

-r rec.pdb
-r = receptor 파일 지정 옵션.
rec.pdb 파일이 도킹 대상이 되는 수용체 단백질 구조야.

-l lig.sdf
-l = ligand 파일 지정 옵션.
lig.sdf는 실제로 도킹하려는 리간드(작은 분자) 구조야.

--autobox_ligand orig.sdf
도킹할 때 리간드를 넣을 결합 포켓(binding site) 범위를 자동으로 정의하기 위해 사용돼.
orig.sdf는 이미 결합한 리간드 구조 → 이걸 기준으로 그 주변 박스(box)를 만들어 도킹 영역을 자동으로 잡아줘.
즉, “원래 리간드가 있던 자리”를 중심으로 새로운 리간드를 도킹하게 되는 거지.

-o docked.sdf.gz
-o = 출력 파일 지정 옵션.
도킹 결과(예상 결합 구조와 포즈들)를 압축된 SDF 파일(.sdf.gz)로 저장해.

여러 포즈가 저장될 수도 있어.

✨ 요약하면:
👉 lig.sdf 리간드를 rec.pdb 단백질의 결합 자리(기존 orig.sdf 리간드가 있던 위치)에 도킹하고, 결과를 docked.sdf.gz 파일에 저장한다.

To dock ligand `lig.sdf` to a binding site on `rec.pdb` defined by another ligand `orig.sdf`:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf -o docked.sdf.gz
```
## 2
🔍 한 줄씩 설명
gnina
Gnina 프로그램 실행. (AutoDock Vina + deep learning scoring)

-r rec.pdb
수용체 단백질 구조 파일 지정.

-l lig.sdf
도킹할 새로운 리간드 파일 지정.

--autobox_ligand orig.sdf
orig.sdf 리간드가 차지했던 자리(결합 포켓)를 기준으로 도킹 박스 자동 생성.

--flexdist_ligand orig.sdf
orig.sdf 주위의 아미노산 잔기들을 "유연(flexible)"하게 취급하라는 옵션.
즉, 단백질의 일부 사이드체인(residues)이 rigid(고정)되지 않고 움직일 수 있게 함.

--flexdist 3.5
orig.sdf 리간드로부터 3.5 Å 이내에 있는 단백질 잔기들의 사이드체인을 flexible하게 둔다.
보통 rigid docking이 일반적인데, 결합 포켓이 매우 유연하다는 근거가 있을 때만 쓰는 게 권장됨.
왜냐면 search space가 훨씬 커지고 결과가 불안정해질 수 있기 때문이야.

-o flex_docked.sdf.gz
도킹 결과(리간드 포즈 + flexible residue 반영)를 압축된 SDF 파일로 저장.

✨ 요약하면:
👉 lig.sdf 리간드를 rec.pdb에 도킹하되, orig.sdf 주변 3.5 Å 안쪽의 단백질 사이드체인들을 유연하게 움직일 수 있도록 허용해서 결과(flex_docked.sdf.gz)를 저장한다.

🆕 새 옵션
--flexdist_ligand orig.sdf
기준 리간드(orig.sdf)를 중심으로 flexible residue 탐색을 시작한다는 뜻.
즉, rigid하게 고정된 단백질이 아니라 해당 리간드 주변 아미노산 사이드체인이 도킹 중에 움직일 수 있음.

--flexdist 3.5
flexible residue로 취급할 거리 cutoff 설정.
여기서는 orig.sdf 주변 3.5 Å 안쪽 잔기들이 flexible로 지정됨.
값이 커질수록 더 많은 잔기가 유연해지고 연산 복잡도도 커짐.

🧭 언제 쓰면 좋은가?
좋은 상황
포켓이 유연하다고 알려져 있는 경우
예: induced-fit 효과가 큰 효소, loop가 유연한 pocket, 기존 rigid docking으로 docking pose 예측이 전혀 맞지 않을 때.

안 좋은 상황 (추천하지 않는 경우)
포켓이 rigid한 단백질 (예: 깊고 잘 정의된 cavity)
데이터가 부족해서 유연성을 강제로 주면 오히려 잡음이 늘고 결과가 불안정해짐.
계산량이 많아져 시간이 훨씬 오래 걸림.

즉, 이 명령어는
👉 "내 단백질의 결합 부위가 딱딱하지 않고, 리간드가 들어오면서 주변 잔기들이 움직여야 한다고 확실히 예상되는 경우"
에만 쓰는 게 좋아.

To perform docking with flexible sidechain residues within 3.5 Angstroms of `orig.sdf` (generally not recommend unless prior knowledge indicates pocket is highly flexible):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --flexdist_ligand orig.sdf --flexdist 3.5 -o flex_docked.sdf.gz
```

## 3
🆕 새 옵션
--autobox_ligand rec.pdb
보통은 이미 결합해 있는 리간드(orig.sdf)를 기준으로 box를 잡는데, 여기서는 **리간드 대신 단백질 전체(rec.pdb)**를 넣었어.
그 결과 → 결합 포켓을 특정하지 않고 단백질 전체를 탐색 공간으로 삼아 도킹을 수행함.
즉, “어디에 붙을지 모르는 상황”에서 리간드가 단백질 전체 표면을 대상으로 결합할 수 있는 위치를 찾는 것.

--exhaustiveness 64
도킹 탐색(search)의 철저함 정도를 설정하는 옵션 (기본값은 보통 8).
값이 클수록 → 탐색을 훨씬 많이 해서 더 다양한 포즈를 탐색하지만, 계산 시간이 길어짐.
64는 꽤 높은 값이라 정확도↑, 속도↓ 트레이드오프를 의미해.

🧭 언제 쓰면 좋은가?
좋은 상황
단백질의 **결합 부위(binding site)**를 전혀 모를 때.
구조 기반 drug discovery 초기 단계 → “이 리간드가 단백질 어디에 붙을까?”를 알아보는 exploratory docking.
단백질에 여러 potential binding sites가 있을 것으로 의심될 때.

덜 좋은 상황
이미 binding site가 알려져 있을 때 → 굳이 whole docking을 하면 search space가 커지고 noise가 많아짐.
계산량이 크게 늘어나서 시간 소모가 심함.

👉 요약: 이 명령어는 **“binding site를 모르는 상황에서 단백질 전체를 대상으로 리간드가 어디 붙을지 찾아보는 탐색용”**으로 쓰는 게 좋아.

To perform whole protein docking:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand rec.pdb -o whole_docked.sdf.gz --exhaustiveness 64
```

## 4
🆕 새 옵션
--cnn_scoring refinement
보통 Gnina는 도킹을 하고 나서 rescore 모드(딥러닝 CNN 점수로 포즈 재평가)만 수행하는 게 기본이야.
그런데 refinement 모드를 주면, 에너지 최소화 과정(리간드 위치 최적화) 에서부터 CNN 점수를 직접 사용해.
즉, docking 과정의 마지막 "세밀한 위치 조정(refinement)" 단계에서 CNN이 계속 개입해서 더 정밀하게 최적화함.
단점: 계산량이 엄청 늘어서 기본 rescore 대비 약 10배 느려짐.

🧭 언제 쓰면 좋은가?
좋은 상황
리간드 포즈 정확도가 정말 중요한 경우.
후보 리간드 수가 많지 않고, 한두 개 리간드를 정밀하게 dock할 때.
기존 rigid docking 결과가 애매하고, 포즈 예측 개선이 꼭 필요할 때.

덜 좋은 상황
수천 개 이상의 라이브러리를 high-throughput screening 하는 경우 → 너무 느려서 비효율적.
빠른 초기 스크리닝 단계.

👉 정리하면:
--cnn_scoring refinement는 “빠른 스크리닝이 아니라, 중요한 리간드 몇 개를 고정밀 refine할 때” 쓰는 고급 옵션이야. 🚀

To utilize the default ensemble CNN in the energy minimization during the refinement step of docking (10 times slower than the default rescore option):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn_scoring refinement -o cnn_refined.sdf.gz
```

## 5
🆕 새 옵션
--cnn_scoring all
도킹의 모든 단계(sampling, refinement, rescoring)에 CNN 점수를 직접 사용.
즉, 랜덤 탐색부터 최종 포즈 평가까지 전부 deep learning scoring으로 컨트롤하는 방식.
결과적으로 가장 정밀한 CNN 기반 도킹이 가능하지만…
기본 rescore 모드보다 약 1000배 느림 ⚠️

🧭 언제 쓰면 좋은가?
상황	설명
✅ 좋은 상황	- 단일 리간드(or 소수 리간드)에 대해 최고 수준 정밀도가 필요할 때
- 연구 목적상 CNN의 full potential을 평가하려고 할 때
⚠️ 나쁜 상황	- 수십~수천 개 라이브러리를 screening할 때 (실질적으로 불가능할 만큼 느림)
- 초기 탐색 단계에서 빠른 결과가 필요한 경우

👉 정리하면:
--cnn_scoring all은 **“극한의 정밀도를 원할 때 쓰는 옵션”**이야. 하지만 현실적으로는 너무 느려서 보통 연구용 실험(benchmark)이나 중요 리간드 소수 대상으로만 사용해.
To utilize the default ensemble CNN for every step of docking (1000 times slower than the default rescore option):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn_scoring all -o cnn_all.sdf.gz
```

## 6
🆕 새 옵션
--scoring vinardo
도킹 시 사용하는 경험적(empirical) scoring function을 Vinardo로 설정.
Vinardo는 AutoDock Vina의 scoring을 개선한 버전으로, 수소 결합·소수성 상호작용 같은 항목을 더 세밀히 반영해.
보통 Vina보다 **포즈 예측 정확도와 점수 상관성(correlation)**이 개선된 것으로 알려져 있어.

--cnn_scoring none
딥러닝 기반 CNN 점수는 사용하지 않겠다는 뜻.
즉, 순수하게 경험적 scoring function(Vinardo)만 사용해서 도킹을 수행함.

🧭 언제 쓰면 좋은가?
상황	설명
✅ 좋은 상황	- baseline 비교: CNN 기반 스코어와 경험적 스코어의 차이를 보고 싶을 때.
- 딥러닝 모델을 신뢰하기 어렵거나, 기존 Vina/Vinardo 기반 결과와 일관성을 확인하고 싶을 때.
- 빠른 계산이 필요할 때 (CNN 없이 가볍게 실행).
- 
⚠️ 나쁜 상황	- 최신 CNN scoring이 더 신뢰도 높은 상황 (예: 학습된 모델이 해당 단백질-리간드 domain에 잘 맞을 때).
- 정밀 refinement가 필요한 경우 (Vinardo만 쓰면 최신 CNN 방식보다 덜 정확할 수 있음).

👉 요약:
이 명령어는 “CNN을 완전히 끄고, Vinardo만 써서 경험적 점수 기반 도킹을 하고 싶을 때” 사용하는 옵션이야. 보통 CNN 결과랑 비교용으로 많이 쓰이지!
To utilize all empirical scoring using the [Vinardo](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155183) scoring function:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --scoring vinardo --cnn_scoring none -o vinardo_docked.sdf.gz
```

## 7
🆕 새 옵션
--cnn dense
도킹 과정에서 사용할 CNN 모델을 기본 ensemble 말고 다른 특정 CNN 아키텍처로 지정하는 옵션.
여기서는 dense라는 모델을 선택한 거야.
gnina --help를 보면 dense, general_default2018, crossdock_default2018 같은 다양한 사전학습 CNN들이 있어.
각 CNN은 학습된 데이터셋과 네트워크 구조가 달라서, 특정 상황에서 더 잘 맞을 수 있어.

🧭 언제 쓰면 좋은가?
상황	설명
✅ 좋은 상황	- 특정 CNN이 내 단백질-리간드 계열에 더 잘 맞는다고 알려진 경우.
- 여러 CNN을 써보고 모델별 점수 차이/성능 비교를 하고 싶을 때.
- 논문 reproducibility (예: 어떤 연구에서 dense CNN을 사용했다고 명시한 경우).
- 
⚠️ 나쁜 상황	- CNN 모델에 대해 사전 지식이 전혀 없고, 그냥 default ensemble이 더 안전한 경우.
- 대규모 screening (모델별 차이를 다 돌려보기에는 시간이 오래 걸림).

👉 정리하면:
--cnn dense는 “기본 CNN 말고 특정 CNN 아키텍처를 지정해서 도킹하고 싶을 때” 쓰는 거야. 주로 실험 비교용이나 reproducibility에 유용하지!

To utilize a different CNN during docking (see help for possible options):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn dense -o dense_docked.sdf.gz
```

## 8
🆕 새 옵션
--minimize
리간드가 이미 단백질의 결합부위에 배치된 상태라고 가정하고, 도킹 탐색은 하지 않고 바로 에너지 최소화(geometry optimization) 만 수행.
즉, 리간드의 포즈를 크게 바꾸지 않고, 국소적으로 위치와 결합 형태를 더 안정화시키는 과정이야.
결과는 최적화된 포즈 + 점수가 함께 출력됨.

🧭 언제 쓰면 좋은가?
상황	설명
✅ 좋은 상황	- 리간드 포즈가 이미 알려져 있을 때 (예: co-crystal 구조, 수동으로 배치한 경우).
- docking 없이 기존 포즈의 에너지 평가만 하고 싶을 때.
- docking 결과를 더 정제(refine)해서 안정화시키고 싶을 때.
- 
⚠️ 나쁜 상황	- 아직 binding site가 확실하지 않거나, 리간드가 제자리를 못 잡은 경우 → 단순 최소화만 하면 잘못된 포즈가 그대로 유지됨.
- 새로운 결합 모드를 찾고 싶은 경우 (탐색을 전혀 하지 않음).

👉 요약하면:
이 명령어는 “이미 binding site에 들어간 리간드들을 그대로 두고, 포즈를 최적화 + 스코어만 매기고 싶을 때” 쓰는 거야. 🚀

To minimize and score ligands `ligs.sdf` already positioned in a binding site:
```
gnina -r rec.pdb -l ligs.sdf --minimize -o minimized.sdf.gz
```

## 9
🆕 새 옵션
--covalent_rec_atom A:601:FE
수용체에서 공유 결합(covalent bond)을 형성할 원자를 지정.
여기서는 chain A의 residue 번호 601에 있는 철(Fe) 원자를 대상으로 함.
즉, docking 과정에서 리간드가 이 철 원자와 결합할 수 있도록 강제하는 거야.

--covalent_lig_atom_pattern '[$(n1nccc1)]'
리간드에서 결합할 특정 원자 패턴을 SMARTS 형식으로 지정.
'[$(n1nccc1)]'는 pyrazole 고리 안의 **질소 원자(N)**를 의미해.
따라서 리간드의 pyrazole N과 수용체의 Fe가 공유결합을 형성하도록 도킹이 진행됨.
여기서
'[$(n1nccc1)]' = pyrazole 고리(n1nccc1)의 질소 원자 SMARTS 패턴.
따라서 pyrazole N이 covalent 결합 대상이 되는 거야.

👉 다른 원자로 바꾸는 방법
이 패턴을 다른 SMARTS 표현식으로 바꾸면, 해당 원자가 covalent 결합 지점으로 인식돼.

예시:
원자	SMARTS 패턴	의미
아민 N (-NH2)	[NH2]	1차 아민 질소
하이드록실 O (-OH)	[OH]	알코올 산소
티올 S (-SH)	[SH]	황 원자
방향족 N (pyridine N)	[nH0]	방향족 질소(수소 없는 경우)
특정 탄소	[C;H2]	2개의 수소를 가진 탄소 (sp³ methylene)

예를 들어, -OH 그룹 산소를 covalent target으로 쓰고 싶다면:
--covalent_lig_atom_pattern '[OH]'


이렇게 바꿀 수 있어.

⚠️ 주의할 점
SMARTS 패턴이 리간드 파일(lig.sdf or conformer.sdf.gz) 안에 실제로 존재하는 원자와 매칭돼야 해.
안 맞으면 gnina가 covalent atom을 못 찾고 에러가 날 수 있어.
covalent_rec_atom 쪽과 화학적으로 어울리는 패턴을 써야 안정적인 결합 모델링이 돼 (예: 금속-질소, 금속-황, 시스테인-SH 등).

👉 정리하면:
네, SMARTS 패턴을 바꾸면 리간드의 다른 원자를 covalent 결합 지점으로 지정할 수 있어! 🔧
To covalently dock a pyrazole to a specific iron atom on the receptor with the bond formed between a nitrogen of the pyrazole and the iron.

전체적으로 🧭 언제 쓰면 좋은가?
상황	설명
✅ 좋은 상황	- 금속 착물(metalloenzyme) 시스템 연구 (예: heme, iron-sulfur cluster, zinc-binding proteins).
- covalent inhibitor 설계 (특정 포켓에서 리간드가 단백질/금속과 공유결합하도록 유도).
- 일반 도킹으로는 포착하기 어려운 금속-리간드 상호작용을 정확히 모델링하고 싶을 때.
  
⚠️ 나쁜 상황	- 표준 비공유적(non-covalent) 도킹을 하고 싶을 때.
- 결합 부위가 불확실하거나, covalent binding evidence가 없는 경우 → 잘못된 강제 결합이 오히려 왜곡된 결과를 낳을 수 있음.

👉 요약:
이 명령어는 “리간드의 pyrazole N이 수용체의 특정 Fe 원자와 반드시 공유결합을 형성하게” 해서 도킹을 수행하는 옵션이야. ⚡
To covalently dock a pyrazole to a specific iron atom on the receptor with the bond formed between a nitrogen of the pyrazole and the iron.
```
gnina  -r rec.pdb.gz -l conformer.sdf.gz --autobox_ligand bindingsite.sdf.gz --covalent_rec_atom A:601:FE --covalent_lig_atom_pattern '[$(n1nccc1)]' -o output.sdf.gz 
```

## 10
--covalent_lig_atom_position x,y,z
공유결합을 형성할 리간드 측 원자를 OpenBabel 자동 휴리스틱 대신, 네가 지정한 좌표(Å) 에 직접 배치해.
좌표계는 수용체(rec.pdb.gz)의 좌표 프레임과 동일해야 해(파일마다 원점·단위가 같아야 함).
목적: 금속–질소 같은 정해진 결합 거리/방향성을 정확히 맞추고 시작하고 싶을 때.

--covalent_optimize_lig
위에서 배치한 리간드(혹은 기존 배치)를 UFF(Universal Force Field) 로 국소 최적화해서, 초기 배치에서 생길 수 있는 비현실적 거리·각도를 완화해.
단백질은 그대로 두고 리간드 단독을 UFF로 다듬는 개념(리간드/잔기 복합체의 기하 안정화).

🧭 언제 쓰면 좋은가?
좋은 상황
금속 배위 거리/각(예: Fe–N ≈ 2.0 Å대)을 정밀하게 맞춰 시작하고 싶을 때.
자동 배치가 계속 엇나가서 수동으로 기준점을 확실히 잡고 싶을 때.
코어 포즈는 확신하지만, 말단 토션만 가볍게 정리(UFF) 하고 싶을 때.

덜 좋은 상황
결합 지점 좌표에 자신이 없을 때(잘못 찍으면 전 과정이 비틀어짐).
포켓 자체 재배치/유연화가 필요한 문제(이 옵션은 리간드 중심이라 단백질 재배열은 안 함).

✅ 실무 팁(짧게)
좌표는 Å 단위, 수용체와 동일 프레임인지 꼭 확인.

--covalent_lig_atom_pattern이 단 하나의 원자만 매칭되도록 SMARTS를 명확히.
필요하면 후속으로 --minimize나 CNN 기반 refine(예: --cnn_scoring refinement)로 미세 조정까지 고려해봐.

The same as above, but with the covalently bonding ligand atom manually positioned (instead of using OpenBabel binding heuristics) and the ligand/residue complex UFF optimized.
```
gnina  -r rec.pdb.gz -l conformer.sdf.gz --autobox_ligand bindingsite.sdf.gz --covalent_lig_atom_position -11.796,31.887,72.682  --covalent_optimize_lig  --covalent_rec_atom A:601:FE --covalent_lig_atom_pattern '[$(n1nccc1)]' -o output.sdf.gz 
```


## 11.📘 Gnina 주요 옵션 한국어 설명 정리

| 옵션 | 의미 (한국어) | 언제 쓰면 좋은가 |
|------|--------------|------------------|
| **`-r rec.pdb`** | 도킹할 수용체(단백질) 구조 지정 | 항상 필요 |
| **`-l lig.sdf`** | 도킹할 리간드(소분자) 구조 지정 | 항상 필요 |
| **`--autobox_ligand file.sdf`** | 특정 리간드 주위로 도킹 박스 자동 생성 | 결합 포켓이 이미 알려져 있을 때 |
| **`--flexdist_ligand file.sdf`** | 기준 리간드 주변 잔기를 flexible 처리 | 포켓이 유연하다고 알려진 경우 |
| **`--flexdist 3.5`** | 기준 리간드에서 3.5 Å 이내 잔기들만 flexible | induced-fit 가능성이 있을 때 |
| **`--exhaustiveness 64`** | 탐색 철저도(기본값 8, 높을수록 느리지만 정확) | 포즈 정확도 높이고 싶을 때 |
| **`--cnn_scoring rescore`** *(기본)* | 도킹 후 CNN 점수로 재평가만 함 | 빠른 고속 스크리닝 |
| **`--cnn_scoring refinement`** | refinement 단계에서 CNN 반영 (10배 느림) | 소수 리간드의 정밀 포즈 필요 |
| **`--cnn_scoring all`** | 모든 단계에서 CNN 반영 (1000배 느림) | 최고 정밀도 연구용 |
| **`--scoring vinardo`** | 경험적 scoring function Vinardo 사용 | CNN과 비교, baseline 용 |
| **`--cnn_scoring none`** | CNN 사용 안 하고 경험적 점수만 사용 | CNN이 불확실할 때 |
| **`--cnn dense`** | 기본 앙상블 대신 특정 CNN 사용 | 논문 재현, CNN 비교 실험 |
| **`--minimize`** | 이미 배치된 리간드를 에너지 최소화 | co-crystal 포즈 정제, 평가 |
| **`--covalent_rec_atom A:601:FE`** | 수용체의 특정 원자와 covalent bond 지정 | 금속·특정 잔기와 공유결합 모델링 |
| **`--covalent_lig_atom_pattern 'SMARTS'`** | 리간드에서 결합할 원자 패턴 지정 | 특정 작용기 원자 결합 |
| **`--covalent_lig_atom_position x,y,z`** | 리간드 결합 원자 좌표 직접 지정 | 자동 배치 대신 수동 위치 지정 |
| **`--covalent_optimize_lig`** | 리간드 구조를 UFF로 국소 최적화 | covalent 초기 포즈 안정화 |

---

### ✨ 요약
- **기본 옵션** → `-r`, `-l`, `--autobox_ligand`  
- **정밀도 제어** → `--exhaustiveness`, `--cnn_scoring` 계열  
- **스코어링 방식** → `--scoring vinardo`, `--cnn_scoring none/all/refinement`  
- **리간드 처리** → `--minimize`, `--flexdist*`  
- **공유결합 도킹** → `--covalent_*` 계열  


# All options:
## 📥 Input Options (입력 관련 옵션)

| 옵션 | 의미 (원문) | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|-------------|------------------|
| **`-r, --receptor`** | rigid part of the receptor | 도킹할 수용체(단백질)의 고정된 부분 구조 파일 | 항상 필요 |
| **`--flex`** | flexible side chains, if any (PDBQT) | 별도로 준비된 flexible side chain 구조(PDBQT 형식) | advanced 사용자: 직접 flexible residue 정의할 때 |
| **`-l, --ligand`** | ligand(s) | 도킹할 리간드 파일 (여러 개 가능) | 항상 필요 |
| **`--flexres chain:resid,...`** | flexible side chains specified by comma separated list of chain:resid | `체인:잔기번호` 리스트로 flexible residue 직접 지정 (쉼표로 구분) | 특정 잔기만 유연하게 처리하고 싶을 때 |
| **`--flexdist_ligand file.sdf`** | Ligand to use for flexdist | 기준 리간드 파일을 사용해 flexible residue 범위 계산 | binding site 유연성을 반영하고 싶을 때 |
| **`--flexdist distance`** | set all side chains within specified distance to flexdist_ligand to flexible | 기준 리간드에서 지정 거리(Å) 안에 있는 모든 side chain을 flexible 처리 | induced-fit 의심될 때 |
| **`--flex_limit N`** | Hard limit for the number of flexible residues | flexible residue 개수의 상한선 지정 | 너무 많은 잔기가 flexible 되는 걸 방지 |
| **`--flex_max N`** | Retain at most the closest flex_max flexible residues | 가장 가까운 flexible residue N개만 유지 | 연산량을 줄이고 싶을 때 |

```
Input:
  -r [ --receptor ] arg              rigid part of the receptor
  --flex arg                         flexible side chains, if any (PDBQT)
  -l [ --ligand ] arg                ligand(s)
  --flexres arg                      flexible side chains specified by comma 
                                     separated list of chain:resid
  --flexdist_ligand arg              Ligand to use for flexdist
  --flexdist arg                     set all side chains within specified 
                                     distance to flexdist_ligand to flexible
  --flex_limit arg                   Hard limit for the number of flexible 
                                     residues
  --flex_max arg                     Retain at at most the closest flex_max 
                                     flexible residues
```


## 🔍 Search Space Options (탐색 공간 관련 옵션)

| 옵션 | 의미 (원문) | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|-------------|------------------|
| **`--center_x`** | X coordinate of the center | 탐색 박스 중심의 X 좌표 | 결합 포켓 중심 좌표를 직접 지정할 때 |
| **`--center_y`** | Y coordinate of the center | 탐색 박스 중심의 Y 좌표 | 위와 동일 |
| **`--center_z`** | Z coordinate of the center | 탐색 박스 중심의 Z 좌표 | 위와 동일 |
| **`--size_x`** | size in the X dimension (Å) | 탐색 박스의 X축 크기 (Å) | 결합 포켓 크기를 수동으로 지정할 때 |
| **`--size_y`** | size in the Y dimension (Å) | 탐색 박스의 Y축 크기 (Å) | 위와 동일 |
| **`--size_z`** | size in the Z dimension (Å) | 탐색 박스의 Z축 크기 (Å) | 위와 동일 |
| **`--autobox_ligand file.sdf`** | Ligand to use for autobox | 리간드 주변으로 자동 탐색 박스 생성 | co-crystal 리간드 기반으로 box 자동 설정 |
| **`--autobox_add N`** | Amount of buffer space to add (default +4 Å) | 자동 생성된 박스의 여유 공간 (기본 4 Å) | 박스를 조금 더 크게 설정하고 싶을 때 |
| **`--autobox_extend` (기본=1)** | Expand autobox if ligand cannot rotate | 리간드가 자유롭게 회전할 수 있도록 박스를 자동 확장 | 긴 리간드나 유연한 구조에 유용 |
| **`--no_lig`** | no ligand; for sampling/minimizing flexible residues | 리간드 없이 실행 (flexible residue 샘플링/최소화 전용) | 리간드 도킹 없이 수용체만 다루고 싶을 때 |

### 📐 Search Space 개념도  

탐색 박스(Search Box)는 **리간드가 탐색되는 3차원 공간**을 정의합니다.  
박스는 중심 좌표(`center_x, center_y, center_z`)와 각 방향 크기(`size_x, size_y, size_z`)로 설정됩니다.  
```
       Z
       ↑
       │
       │      ┌───────────────┐
       │     /               /│
       │    /               / │
       │   /               /  │  size_y
       │  └───────────────┘   │
       │  │               │   │
       │  │               │   │
       │  │               │   │
       │  └───────────────┘───┼───→  X
      center (x,y,z)           │
                               
             size_x
```
- **center_x, center_y, center_z** → 박스 중심 좌표  
- **size_x, size_y, size_z** → 박스의 각 축 크기 (Å 단위)  

```
Search space (required):
  --center_x arg                     X coordinate of the center
  --center_y arg                     Y coordinate of the center
  --center_z arg                     Z coordinate of the center
  --size_x arg                       size in the X dimension (Angstroms)
  --size_y arg                       size in the Y dimension (Angstroms)
  --size_z arg                       size in the Z dimension (Angstroms)
  --autobox_ligand arg               Ligand to use for autobox. A multi-ligand 
                                     file still only defines a single box.
  --autobox_add arg                  Amount of buffer space to add to 
                                     auto-generated box (default +4 on all six 
                                     sides)
  --autobox_extend arg (=1)          Expand the autobox if needed to ensure the
                                     input conformation of the ligand being 
                                     docked can freely rotate within the box.
  --no_lig                           no ligand; for sampling/minimizing 
                                     flexible residues
```


## 🔗 Covalent Docking Options (공유결합 도킹 옵션)

| 옵션 | 의미 (원문) | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|-------------|------------------|
| **`--covalent_rec_atom arg`** | Receptor atom ligand is covalently bound to. 지정: `chain:resnum:atom_name` 또는 `x,y,z` 좌표 | 공유결합이 형성될 수용체의 특정 원자 지정 (예: `A:601:FE` 또는 좌표) | 금속 착물, 특정 아미노산 잔기 등 covalent 결합 목표가 명확할 때 |
| **`--covalent_lig_atom_pattern arg`** | SMARTS expression for ligand atom | 리간드에서 공유결합을 형성할 원자 SMARTS 패턴 (예: `'[SH]'`, `'[OH]'`, `'[$(n1nccc1)]'`) | 결합할 리간드 원자를 특정하고 싶을 때 |
| **`--covalent_lig_atom_position x,y,z`** | Initial placement of ligand atom (Å). 기본: OpenBabel 자동 배치 | 공유결합 리간드 원자의 초기 좌표를 직접 지정 | 자동 배치가 마음에 안 들고, 결합 거리·각을 수동으로 설정하고 싶을 때 |
| **`--covalent_fix_lig_atom_position`** | Fix ligand atom at specified position | `--covalent_lig_atom_position`에서 지정한 좌표를 초기 구조가 아니라 **고정 위치**로 사용 | 리간드 원자를 특정 좌표에 완전히 고정하고 싶을 때 |
| **`--covalent_bond_order N`** (기본=1) | Bond order of covalent bond | 공유결합의 결합 차수 설정 (단일/이중/삼중결합) | 금속 배위나 특수한 결합을 시뮬레이션할 때 |
| **`--covalent_optimize_lig`** | Optimize ligand-residue complex with UFF | 공유결합된 리간드/잔기 복합체를 UFF force field로 최적화 → 결합 거리·각 조정 | covalent docking 후 구조 안정화 필요할 때 |

### 🧪 Common SMARTS Patterns for Covalent Docking (자주 쓰이는 SMARTS 패턴)

| SMARTS 패턴 | 의미 (한국어) | 예시 상황 |
|-------------|--------------|-----------|
| **`[SH]`** | 티올(-SH) 그룹의 황 원자 | 시스테인 잔기와 공유결합 형성 (예: covalent inhibitor targeting Cys) |
| **`[OH]`** | 하이드록실(-OH) 그룹의 산소 원자 | 세린/트레오닌 잔기, 금속-산소 결합 |
| **`[NH2]`** | 아민(-NH₂) 질소 원자 | 리신(Lys) side chain과 공유결합 |
| **`[nH0]`** | 방향족 고리 안의 질소(수소 없음) | 피리딘 N, 금속 착물과의 배위 결합 |
| **`[$(n1nccc1)]`** | 피라졸(pyrazole) 고리 내 질소 | 철(Fe) 등 금속과 배위 결합 |
| **`[C;H2]`** | 수소 2개를 가진 sp³ 탄소 (메틸렌) | 드물지만 특수한 탄소-금속 결합 모델링 |
| **`[Cl]`** | 염소 원자 | 금속-할로겐 상호작용 모델링 |
| **`[Br]`** | 브로민 원자 | 위와 동일 |

```
Covalent docking:
  --covalent_rec_atom arg            Receptor atom ligand is covalently bound 
                                     to.  Can be specified as 
                                     chain:resnum:atom_name or as x,y,z 
                                     Cartesian coordinates.
  --covalent_lig_atom_pattern arg    SMARTS expression for ligand atom that 
                                     will covalently bind protein.
  --covalent_lig_atom_position arg   Optional.  Initial placement of covalently
                                     bonding ligand atom in x,y,z Cartesian 
                                     coordinates.  If not specified, 
                                     OpenBabel's GetNewBondVector function will
                                     be used to position ligand.
  --covalent_fix_lig_atom_position   If covalent_lig_atom_position is 
                                     specified, fix the ligand atom to this 
                                     position as opposed to using this position
                                     to define the initial structure.
  --covalent_bond_order arg (=1)     Bond order of covalent bond. Default 1.
  --covalent_optimize_lig            Optimize the covalent complex of ligand 
                                     and residue using UFF. This will change 
                                     bond angles and lengths of the ligand.
```


## ⚖️ Scoring & Minimization Options (점수 함수 및 최소화 옵션)

| 옵션 | 의미 (원문) | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|-------------|------------------|
| **`--scoring arg`** | specify alternative built-in scoring function | 내장된 다른 스코어링 함수 지정 (`ad4_scoring`, `vina`, `vinardo`, etc.) | CNN과 비교하거나 baseline 계산할 때 |
| **`--custom_scoring file`** | custom scoring function file | 사용자 정의 스코어링 함수 파일 지정 | 직접 점수 함수를 개발했을 때 |
| **`--custom_atoms file`** | custom atom type parameters file | 사용자 정의 원자 타입 파라미터 파일 지정 | 새로운 원자 타입을 다룰 때 |
| **`--score_only`** | score provided ligand pose | 주어진 리간드 포즈 점수만 계산 (도킹 없음) | co-crystal 구조 평가 |
| **`--local_only`** | local search only using autobox | 오토박스를 기반으로 국소 탐색만 수행 | 빠른 테스트, 보통 `--minimize` 추천 |
| **`--minimize`** | energy minimization | 에너지 최소화 수행 | 리간드 포즈 정제 |
| **`--randomize_only`** | generate random poses | 무작위 포즈 생성 (충돌 피하려 시도) | 랜덤 초기화 실험 |
| **`--num_mc_steps N`** | fixed number of MC steps per chain | 각 몬테카를로 체인에서 고정된 단계 수 | 샘플링 깊이 제어 |
| **`--max_mc_steps N`** | cap on number of MC steps | MC 최대 단계 제한 | 과도한 계산 방지 |
| **`--num_mc_saved N`** | number of top poses saved | 체인별 상위 포즈 개수 저장 | 다양한 포즈 보존 |
| **`--temperature T`** | temperature for metropolis criterion | Metropolis 기준 온도 | 수용/거부 확률 조정 |
| **`--minimize_iters N`** | number of steepest descent iterations | 최소화 반복 횟수 (기본은 rotors 수 기반) | 더 정밀한 최소화 필요할 때 |
| **`--accurate_line`** | use accurate line search | 더 정밀한 라인 탐색 사용 | 최소화 정확도↑, 속도↓ |
| **`--simple_ascent`** | use simple gradient ascent | 단순 그래디언트 상승법 사용 | 실험적 최적화 |
| **`--minimize_early_term`** | early stop minimization | 수렴 전 최소화 조기 종료 | 계산 시간 절약 |
| **`--minimize_single_full`** | single full minimization | 전체 최소화 1회 수행 (사전 평가 없이) | 안정적 최소화 원할 때 |
| **`--approximation mode`** | linear, spline, exact | 근사화 방식 선택 | 속도 vs 정확도 트레이드오프 |
| **`--factor N`** | approximation factor | 근사화 정밀도 조정 (높을수록 세밀) | 더 정확한 평가 필요 시 |
| **`--force_cap N`** | max allowed force | 최대 허용 힘 설정 (낮으면 부드러운 최소화) | clash 완화 |
| **`--user_grid file`** | Autodock map file for user grid | 사용자 지정 그리드 데이터 사용 | 외부 계산결과 활용 |
| **`--user_grid_lambda N`** | scale user_grid and functional scoring | 사용자 그리드와 함수 스코어의 스케일링 | scoring 혼합 |
| **`--print_terms`** | print all terms | 모든 가능한 항목과 기본 파라미터 출력 | 디버깅, 학습용 |
| **`--print_atom_types`** | print all atom types | 사용 가능한 원자 타입 출력 | custom atom 정의 전 확인 |


### ⚖️ Scoring & Minimization Options (점수 함수 및 최소화 옵션) 재분류

####  🔹 Scoring Functions (점수 함수)
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--scoring arg`** | 내장 스코어링 함수 선택 (`vina`, `vinardo`, `ad4_scoring` 등) | CNN과 비교하거나 baseline 점수가 필요할 때 |
| **`--custom_scoring file`** | 사용자 정의 스코어링 함수 파일 지정 | 직접 만든 scoring 함수 적용 |
| **`--custom_atoms file`** | 사용자 정의 원자 타입 파라미터 파일 | 새로운 원자 타입 다룰 때 |
| **`--score_only`** | 주어진 리간드 포즈 점수만 계산 (도킹 없음) | co-crystal 구조 평가 |
| **`--print_terms`** | 사용 가능한 scoring 항목과 기본 파라미터 출력 | 디버깅, 학습용 |
| **`--print_atom_types`** | 사용 가능한 원자 타입 출력 | custom atom 정의 전 확인 |

---

#### 🔹 Minimization (최소화 옵션)
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--minimize`** | 리간드 포즈 에너지 최소화 | 도킹 후 포즈 정제 |
| **`--minimize_iters N`** | steepest descent 반복 횟수 (기본=rotors 수 기반) | 더 정밀한 최소화 필요할 때 |
| **`--accurate_line`** | 더 정밀한 라인 서치 사용 | 최소화 정확도↑, 속도↓ |
| **`--simple_ascent`** | 단순 gradient ascent 사용 | 실험적 최적화 시도 |
| **`--minimize_early_term`** | 수렴 전에 최소화 조기 종료 | 계산 시간 절약 |
| **`--minimize_single_full`** | 전체 최소화 1회 수행 (사전 평가 없이) | 안정적인 최소화 원할 때 |
| **`--force_cap N`** | 최대 허용 힘 제한 (낮으면 부드럽게) | clash 완화 |
| **`--approximation mode`** | 근사화 방식 (`linear`, `spline`, `exact`) | 속도 vs 정확도 조정 |
| **`--factor N`** | 근사화 정밀도 (높을수록 세밀) | 더 정확한 평가 필요 시 |

---

#### 🔹 Monte Carlo Sampling (샘플링 옵션)
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--local_only`** | 오토박스를 기반으로 국소 탐색만 수행 | 빠른 테스트 (보통 `--minimize` 권장) |
| **`--randomize_only`** | 무작위 포즈 생성 (충돌 피하려 시도) | 랜덤 초기화 실험 |
| **`--num_mc_steps N`** | 각 체인에서 고정된 MC 단계 수 | 샘플링 깊이 제어 |
| **`--max_mc_steps N`** | MC 최대 단계 제한 | 계산량 제한 |
| **`--num_mc_saved N`** | 체인별 상위 포즈 저장 개수 | 다양한 포즈 확보 |
| **`--temperature T`** | Metropolis 기준 온도 | 포즈 수용/거부 확률 제어 |

---

#### 🔹 User Grid (사용자 정의 그리드)
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--user_grid file`** | AutoDock map 파일을 이용한 사용자 그리드 기반 계산 | 외부 grid 데이터 활용 |
| **`--user_grid_lambda N`** | 사용자 그리드와 scoring의 스케일링 인자 | 혼합 scoring 실험 |

```
Scoring and minimization options:
  --scoring arg                      specify alternative built-in scoring 
                                     function: ad4_scoring default dkoes_fast 
                                     dkoes_scoring dkoes_scoring_old vina 
                                     vinardo
  --custom_scoring arg               custom scoring function file
  --custom_atoms arg                 custom atom type parameters file
  --score_only                       score provided ligand pose
  --local_only                       local search only using autobox (you 
                                     probably want to use --minimize)
  --minimize                         energy minimization
  --randomize_only                   generate random poses, attempting to avoid
                                     clashes
  --num_mc_steps arg                 fixed number of monte carlo steps to take 
                                     in each chain
  --max_mc_steps arg                 cap on number of monte carlo steps to take
                                     in each chain
  --num_mc_saved arg                 number of top poses saved in each monte 
                                     carlo chain
  --temperature arg                  temperature for metropolis accept 
                                     criterion
  --minimize_iters arg (=0)          number iterations of steepest descent; 
                                     default scales with rotors and usually 
                                     isn't sufficient for convergence
  --accurate_line                    use accurate line search
  --simple_ascent                    use simple gradient ascent
  --minimize_early_term              Stop minimization before convergence 
                                     conditions are fully met.
  --minimize_single_full             During docking perform a single full 
                                     minimization instead of a truncated 
                                     pre-evaluate followed by a full.
  --approximation arg                approximation (linear, spline, or exact) 
                                     to use
  --factor arg                       approximation factor: higher results in a 
                                     finer-grained approximation
  --force_cap arg                    max allowed force; lower values more 
                                     gently minimize clashing structures
  --user_grid arg                    Autodock map file for user grid data based
                                     calculations
  --user_grid_lambda arg (=-1)       Scales user_grid and functional scoring
  --print_terms                      Print all available terms with default 
                                     parameterizations
  --print_atom_types                 Print all available atom types
```


## 🧠 CNN Scoring Options (딥러닝 기반 스코어링 옵션)

### 🔹 CNN Scoring Modes
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cnn_scoring mode`** | CNN 점수 적용 단계 설정<br> - `none`: CNN 사용 안 함<br> - `rescore` (기본): 도킹 후 CNN으로 재평가<br> - `refinement`: refinement 단계에서 CNN 사용 (10배 느림)<br> - `metrorescore`: metropolis + rescore<br> - `metrorefine`: metropolis + refine<br> - `all`: 모든 단계에서 CNN 사용 (1000배 느림) | baseline 비교(`none`), 빠른 스크리닝(`rescore`), 고정밀 포즈(`refinement`), 연구용(`all`) |

---

### 🔹 CNN Model Selection
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cnn model`** | 사용할 내장 CNN 모델 지정<br>(예: `dense`, `general_default2018`, `crossdock_default2018`, `redock_default2018` 등)<br> `PREFIX_ensemble` 형식으로 여러 모델 앙상블 가능 | 논문 재현, 특정 데이터셋 기반 모델 선택, 모델 비교 실험 |
| **`--cnn_model file`** | 사용자 정의 Torch CNN 모델 파일 지정 | 직접 학습한 CNN 모델 적용 |

---

### 🔹 CNN Evaluation Settings
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cnn_rotation N`** | 포즈를 여러 회전(최대 24회)으로 평가 | 포즈 orientation에 민감할 때 정밀 평가 |
| **`--cnn_mix_emp_force`** | CNN과 경험적 force를 합침 | 하이브리드 scoring |
| **`--cnn_mix_emp_energy`** | CNN과 경험적 energy를 합침 | 하이브리드 scoring |
| **`--cnn_empirical_weight W`** | 경험적 force/energy 가중치 설정 (기본=1) | CNN vs 경험적 스코어 비중 조절 |

---

### 🔹 CNN Search Space
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cnn_center_x`** | CNN 평가 중심 X 좌표 | CNN을 특정 위치에서만 집중적으로 평가 |
| **`--cnn_center_y`** | CNN 평가 중심 Y 좌표 | 위와 동일 |
| **`--cnn_center_z`** | CNN 평가 중심 Z 좌표 | 위와 동일 |

---

### 🔹 Debugging
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cnn_verbose`** | CNN 디버깅용 상세 출력 | 모델 성능 분석, 디버깅 필요할 때 |

```
Convolutional neural net (CNN) scoring:
  --cnn_scoring arg (=1)             Amount of CNN scoring: none, rescore 
                                     (default), refinement, metrorescore 
                                     (metropolis+rescore), metrorefine 
                                     (metropolis+refine), all
  --cnn arg                          built-in model to use, specify 
                                     PREFIX_ensemble to evaluate an ensemble of
                                     models starting with PREFIX: 
                                     all_default_to_default_1_3_1 
                                     all_default_to_default_1_3_2 
                                     all_default_to_default_1_3_3 
                                     crossdock_default2018 
                                     crossdock_default2018_1 
                                     crossdock_default2018_1_3 
                                     crossdock_default2018_1_3_1 
                                     crossdock_default2018_1_3_2 
                                     crossdock_default2018_1_3_3 
                                     crossdock_default2018_1_3_4 
                                     crossdock_default2018_2 
                                     crossdock_default2018_3 
                                     crossdock_default2018_4 
                                     crossdock_default2018_KD_1 
                                     crossdock_default2018_KD_2 
                                     crossdock_default2018_KD_3 
                                     crossdock_default2018_KD_4 
                                     crossdock_default2018_KD_5 default1.0 
                                     default2017 dense dense_1 dense_1_3 
                                     dense_1_3_1 dense_1_3_2 dense_1_3_3 
                                     dense_1_3_4 dense_1_3_PT_KD 
                                     dense_1_3_PT_KD_1 dense_1_3_PT_KD_2 
                                     dense_1_3_PT_KD_3 dense_1_3_PT_KD_4 
                                     dense_1_3_PT_KD_def2018 
                                     dense_1_3_PT_KD_def2018_1 
                                     dense_1_3_PT_KD_def2018_2 
                                     dense_1_3_PT_KD_def2018_3 
                                     dense_1_3_PT_KD_def2018_4 dense_2 dense_3 
                                     dense_4 fast general_default2018 
                                     general_default2018_1 
                                     general_default2018_2 
                                     general_default2018_3 
                                     general_default2018_4 
                                     general_default2018_KD_1 
                                     general_default2018_KD_2 
                                     general_default2018_KD_3 
                                     general_default2018_KD_4 
                                     general_default2018_KD_5 
                                     redock_default2018 redock_default2018_1 
                                     redock_default2018_1_3 
                                     redock_default2018_1_3_1 
                                     redock_default2018_1_3_2 
                                     redock_default2018_1_3_3 
                                     redock_default2018_1_3_4 
                                     redock_default2018_2 redock_default2018_3 
                                     redock_default2018_4 redock_default2018_KD
                                     _1 redock_default2018_KD_2 
                                     redock_default2018_KD_3 
                                     redock_default2018_KD_4 
                                     redock_default2018_KD_5
  --cnn_model arg                    torch cnn model file; if not specified a 
                                     default model ensemble will be used
  --cnn_rotation arg (=0)            evaluate multiple rotations of pose (max 
                                     24)
  --cnn_mix_emp_force                Merge CNN and empirical minus forces
  --cnn_mix_emp_energy               Merge CNN and empirical energy
  --cnn_empirical_weight arg (=1)    Weight for scaling and merging empirical 
                                     force and energy 
  --cnn_center_x arg                 X coordinate of the CNN center
  --cnn_center_y arg                 Y coordinate of the CNN center
  --cnn_center_z arg                 Z coordinate of the CNN center
  --cnn_verbose                      Enable verbose output for CNN debugging
```


## 📤 Output Options (출력 관련 옵션)

| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`-o, --out file`** | 출력 파일 이름 지정 (파일 확장자에 따라 형식 결정: SDF, PDBQT 등) | 기본 도킹 결과 저장 |
| **`--out_flex file`** | flexible residue 출력 파일 지정 | flexible docking 시 flexible residue 결과 따로 저장 |
| **`--log file`** | 로그 파일 작성 | 실행 과정 기록이 필요할 때 |
| **`--atom_terms file`** | 원자별 상호작용 값 출력 | per-atom 에너지/상호작용 분석 |
| **`--atom_term_data`** | per-atom 상호작용 데이터를 출력 SDF에 직접 포함 | 결과 파일 안에서 상호작용 값 확인하고 싶을 때 |
| **`--pose_sort_order mode`** | 결과 포즈 정렬 기준 선택<br> - `CNNscore` (기본)<br> - `CNNaffinity`<br> - `Energy` | 원하는 기준으로 결과 포즈 순위를 정렬 |
| **`--full_flex_output`** | flexible residue 출력 시, 전체 구조를 저장 (기본은 flexible residue만 출력) | 전체 수용체 구조까지 포함된 결과가 필요할 때 |

---

👉 요약:

결과 저장 → -o, --out_flex, --log
세부 분석 → --atom_terms, --atom_term_data
결과 정렬 → --pose_sort_order
출력 범위 → --full_flex_output

```
Output:
  -o [ --out ] arg                   output file name, format taken from file 
                                     extension
  --out_flex arg                     output file for flexible receptor residues
  --log arg                          optionally, write log file
  --atom_terms arg                   optionally write per-atom interaction term
                                     values
  --atom_term_data                   embedded per-atom interaction terms in 
                                     output sd data
  --pose_sort_order arg (=0)         How to sort docking results: CNNscore 
                                     (default), CNNaffinity, Energy
  --full_flex_output                 Output entire structure for out_flex, not 
                                     just flexible residues.
```


## ⚙️ Misc Options (기타 옵션)

| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--cpu N`** | 사용할 CPU 개수 지정 (기본: 자동 감지, 실패 시 1) | CPU 자원 제한하거나 멀티코어 활용할 때 |
| **`--seed N`** | 랜덤 시드 고정 | 재현 가능한 결과 필요할 때 |
| **`--exhaustiveness N`** (기본=8) | 전역 탐색 철저도 (시간에 비례) | 정밀 탐색 시 값↑, 빠른 탐색 시 값↓ |
| **`--num_modes N`** (기본=9) | 생성할 최대 결합 포즈 수 | 다양한 포즈 확인하고 싶을 때 |
| **`--min_rmsd_filter X`** (기본=1 Å) | 최종 포즈 중 RMSD가 X Å 이하인 중복 제거 | 비슷한 포즈 중복 제거 |
| **`-q, --quiet`** | 출력 메시지 최소화 | 스크립트 실행 시 로그 간소화 |
| **`--addH`** | 리간드에 자동으로 수소 추가 (기본 ON) | 자동 H 보정이 필요할 때 |
| **`--stripH`** | atom typing 후 극성 H 제거 (기본 OFF, 비극성 H는 항상 제거) | 계산 효율↑, 불필요한 H 제거 |
| **`--device N`** (기본=0) | 사용할 GPU 장치 번호 지정 | 멀티 GPU 환경에서 특정 GPU 선택 |
| **`--no_gpu`** | GPU 가속 비활성화 (강제로 CPU 실행) | GPU 없는 환경, 디버깅 시 |

```
Misc (optional):
  --cpu arg                          the number of CPUs to use (the default is 
                                     to try to detect the number of CPUs or, 
                                     failing that, use 1)
  --seed arg                         explicit random seed
  --exhaustiveness arg (=8)          exhaustiveness of the global search 
                                     (roughly proportional to time)
  --num_modes arg (=9)               maximum number of binding modes to 
                                     generate
  --min_rmsd_filter arg (=1)         rmsd value used to filter final poses to 
                                     remove redundancy
  -q [ --quiet ]                     Suppress output messages
  --addH arg                         automatically add hydrogens in ligands (on
                                     by default)
  --stripH arg                       remove polar hydrogens from molecule 
                                     _after_ performing atom typing for 
                                     efficiency (off by default - nonpolar are 
                                     always removed)
  --device arg (=0)                  GPU device to use
  --no_gpu                           Disable GPU acceleration, even if 
                                     available.
```


## 📝 Configuration & Information Options

### 🔹 Configuration
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--config file`** | 위에서 설명한 모든 옵션들을 설정 파일에 넣어서 불러오기 | 매번 긴 명령어 대신 config 파일로 관리하고 싶을 때 |
```
Configuration file (optional):
  --config arg                       the above options can be put here
```
---

### 🔹 Information
| 옵션 | 한국어 설명 | 언제 쓰면 좋은가 |
|------|-------------|------------------|
| **`--help`** | 기본 사용법 요약 출력 | 명령어 옵션 빠르게 확인할 때 |
| **`--help_hidden`** | 숨겨진 옵션까지 포함한 전체 사용법 출력 | advanced 옵션까지 보고 싶을 때 |
| **`--version`** | 프로그램 버전 표시 | 설치된 gnina 버전 확인 |

```
Information (optional):
  --help                             display usage summary
  --help_hidden                      display usage summary with hidden options
  --version                          display program version
```

CNN Scoring
===========

## 🧠 CNN Scoring Options

`--cnn_scoring` determines at what points of the docking procedure that the CNN scoring function is used.  
`--cnn_scoring` 옵션은 도킹 과정의 어느 단계에서 CNN 기반 점수 함수를 사용할지 결정합니다.

| 모드 (Mode) | 설명 (English) | 설명 (한국어) | 권장 사용 상황 |
|-------------|----------------|---------------|----------------|
| **`none`** | No CNNs used. Uses empirical scoring only. | CNN을 전혀 사용하지 않고, 경험적 스코어링만 사용. | 전통적 baseline 도킹, CNN 영향 배제하고 싶을 때 |
| **`rescore`** *(default)* | CNN used only to rerank final poses. Fastest option. | 도킹 후 최종 포즈만 CNN으로 재정렬. 가장 빠름. | 대규모 라이브러리 고속 스크리닝 |
| **`refinement`** | CNN used to refine poses after MC search and for final ranking. ~10x slower. | Monte Carlo 탐색 후 포즈 refinement 및 최종 랭킹에 CNN 사용. `rescore`보다 10배 느림. | 소수의 리간드 정밀 도킹 |
| **`all`** | CNN used throughout docking. Extremely computationally expensive (~1000x slower). | 도킹 전 과정에서 CNN 사용. 계산량 매우 많음. | 연구용/벤치마킹, 최고 정밀도 필요할 때 |

---

### 🔹 Default CNN Models
- The default CNN scoring function is an ensemble of **3 models** balancing accuracy and speed:  
  - `dense_1_3`  
  - `dense_1_3_PT_KD_3`  
  - `crossdock_default2018_KD_4`  

- GNINA 1.0 default ensemble is also available as **`default1.0`**.  

기본 CNN 스코어링 함수는 **3개 모델 앙상블**로 구성되어 있으며, 정확도와 속도의 균형을 고려하여 선택되었습니다.  
- `dense_1_3`  
- `dense_1_3_PT_KD_3`  
- `crossdock_default2018_KD_4`  

예전 GNINA 1.0의 기본 앙상블도 **`default1.0`** 이름으로 여전히 사용 가능합니다.


Training
========
# 🏋️ Training (모델 학습)

GNINA supports training deep learning models for docking using **PyTorch** (modern) and **Caffe** (legacy).  
GNINA는 도킹을 위한 딥러닝 모델 학습을 **PyTorch(최신)** 와 **Caffe(구버전)** 기반으로 지원합니다.  

---

### 🔹 PyTorch Training
| English | 한국어 |
|---------|--------|
| Scripts for PyTorch GNINA models and pretrained models are available at [gnina-torch](https://github.com/RMeli/gnina-torch). | PyTorch 기반 GNINA 학습 스크립트와 사전 학습된 모델은 [gnina-torch](https://github.com/RMeli/gnina-torch)에서 확인할 수 있습니다. |
| Example code for converting a PyTorch model into a GNINA-usable model file is shown below. Metadata should specify grid resolution, dimension, and atom typing. Defaults are used if not provided. | PyTorch 모델을 GNINA에서 사용 가능한 모델로 변환하는 예시는 아래 코드와 같습니다. metadata에는 그리드 해상도, 차원, 원자 타입 정보가 포함되어야 하며, 없을 경우 기본값이 사용됩니다. |

### 🔹 Caffe Training (Legacy)

| English | 한국어 |
|---------|--------|
| Legacy scripts for training Caffe models can be found at [gnina/scripts](https://github.com/gnina/scripts). | Caffe 기반 학습 스크립트(구버전)는 [gnina/scripts](https://github.com/gnina/scripts)에서 확인할 수 있습니다. |
| Sample pretrained Caffe models are available at [gnina/models](https://github.com/gnina/models). | 예시 pretrained Caffe 모델은 [gnina/models](https://github.com/gnina/models)에서 다운로드할 수 있습니다. |
| ⚠️ Note: Caffe support is legacy and no longer actively developed. | ⚠️ 참고: Caffe 지원은 구버전으로, 더 이상 활발히 개발되지 않습니다. |

---

### 🔹 Datasets

| English | 한국어 |
|---------|--------|
| The **DUD-E docked poses** used in the original GNINA paper can be found [here](http://bits.csb.pitt.edu/files/docked_dude.tar). | 원 논문에서 사용된 **DUD-E docked poses**는 [여기](http://bits.csb.pitt.edu/files/docked_dude.tar)에서 받을 수 있습니다. |
| However, we [do not recommend](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220113) training virtual screening models on DUD-E due to dataset biases. | 하지만 [논문](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220113)에서 언급된 것처럼, **DUD-E는 데이터셋 편향 문제 때문에 virtual screening 학습용으로 권장되지 않습니다.** |
| The **CrossDocked2020 dataset** can be found [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020). | **CrossDocked2020 데이터셋**은 [여기](https://github.com/gnina/models/tree/master/data/CrossDocked2020)에서 받을 수 있습니다. |
| CrossDocked2020 is more suitable for GNINA training and benchmarking as it better reflects real docking scenarios. | CrossDocked2020은 실제 도킹 시나리오를 더 잘 반영하기 때문에 GNINA 학습 및 벤치마킹에 더 적합합니다. |



```python
import torch, json

# Metadata example
d = {
    'resolution': 0.5,
    'dimension' : 23.5,
    'recmap' : '''AliphaticCarbonXSHydrophobe ... Iron''',
    'ligmap': '''AliphaticCarbonXSHydrophobe ... Iron'''
}

extra = {'metadata': json.dumps(d)}

z = torch.zeros((1,28,48,48,48))  # dummy input grid

script = torch.jit.trace(model, z)
script.save('gnina_model.pt', _extra_files=extra)




License
=======

gnina is dual licensed under GPL and Apache.  The GPL license is necessitated by the use of OpenBabel (which is GPL licensed).  In order to use gnina under the Apache license only, all references to OpenBabel must be removed from the source code.
