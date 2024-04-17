# DiCo-NeRF
![drive_6](https://github.com/ziiho08/DiCoNeRF/assets/68531659/24ede675-b83e-4182-b6c5-a3bc5cc0eb9d)
<img width="300" height="250" src="https://github.com/ziiho08/DiCoNeRF/assets/68531659/32dcf84a-b4d7-4d22-a404-d86fda0cf16f"/>


![Fig1](https://github.com/ziiho08/DiCoNeRF/assets/68531659/e25f9d3c-c4b7-4aa0-8d13-65c63d2214ec)
We integrate CLIP embeddings into the NeRF optimization process, which allows us to leverage semantic information provided by CLIP when synthesizing novel views of fisheye driving scenes. The proposed method, DiCo-NeRF, utilizes the distributional differences between the similarity maps obtained from pre-trained CLIP to improve the color field of the NeRF.





## Installation
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). 
Then, clone this repository and run the commands:
```
git clone https://github.com/ziiho08/DiCoNeRF.git
conda activate nerfstudio
cd diconerf/
pip install -e .
ns-install-cli
```

## Training the DiCo-NeRF
To train the DiCo-NeRF, run the command:
```
ns-train diconerf --data [PATH]
```

## Citation
```

```
