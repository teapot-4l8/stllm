```
PS D:\_________________________PythonProject\ST-LLM-Plus-main> cd .\data\
PS D:\_________________________PythonProject\ST-LLM-Plus-main\data> python gen_npz.py --h5_name nyc-bike
Generating training data
(4368, 250, 2)
x shape:  (4345, 12, 250, 2) , y shape:  (4345, 12, 250, 2)
train x:  (3001, 12, 250, 2) y: (3001, 12, 250, 2)
val x:  (672, 12, 250, 2) y: (672, 12, 250, 2)
test x:  (672, 12, 250, 2) y: (672, 12, 250, 2)
```

Based on the code and context in the ChenxiLiu-HNU/ST-LLM repository, the .pkl (pickle) files are primarily used to store and load data objects, especially graph-related data for traffic prediction.

Here’s what the .pkl files do in this repo:

1. **Adjacency Matrices for Graphs:**  
   - Files like ./data/PEMS08/adj_PEMS08.pkl are used to store adjacency matrices representing the spatial connectivity (graph structure) of traffic sensor networks.  
   - These matrices are essential for graph-based deep learning models, which use them to understand the relationships between different nodes (e.g., road segments or sensors) in traffic prediction tasks.

2. **Loading Graph Data:**  
   - In util.py, the function load_graph_data(pkl_filename) loads graph data from a .pkl file. This typically includes:
     - sensor_ids: Identifiers for each sensor/node in the traffic network.
     - sensor_id_to_ind: A mapping from sensor IDs to their indices.
     - adj_mx: The adjacency matrix used by the model.

3. **Conversion and Preprocessing:**  
   - The script pkl.py shows an example of reading a .pkl file with pandas and converting it to a .csv, which indicates the .pkl file holds structured data (like DataFrames) that can be used for further data processing.

**Summary:**  
In this repository, .pkl files are used as containers for important precomputed data, such as adjacency matrices and sensor mappings, which are loaded during model setup and training to enable accurate spatial-temporal modeling for traffic prediction.

## gen_npz.py 

划分训练集的数记得改

```python
    num_test = 672 
    num_val = 672
```


## model_ST_LLM_plus.py

```python
        if num_nodes == 170 or num_nodes == 207:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48  # time interval=30 min
```

```python
self.in_layer = nn.Conv2d(gpt_channel*3, to_gpt_channel, kernel_size=(1, 1))
```

gpt_channel * 3：表示输入数据的通道数。在你的模型中，输入数据由三部分组成：
输入数据本身的特征维度（gpt_channel）。
时间嵌入的特征维度（gpt_channel）。
节点嵌入的特征维度（gpt_channel）。
因此，输入数据的总通道数为 gpt_channel * 3


## model
```
ST_LLM(
  (start_conv): Conv2d(36, 256, kernel_size=(1, 1), stride=(1, 1))
  (Temb): TemporalEmbedding()
  (in_layer): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
  (dropout): Dropout(p=0.1, inplace=False)
  (regression_layer): Conv2d(768, 12, kernel_size=(1, 1), stride=(1, 1))
  (gpt): PFA(
    (gpt2): PeftModel(
      (base_model): LoraModel(
        (model): GPT2Model(
          (wte): Embedding(50257, 768)
          (wpe): Embedding(1024, 768)
          (drop): Dropout(p=0.1, inplace=False)
          (h): ModuleList(
            (0): GPT2Block(
              (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (attn): GPT2Attention(
                (c_attn): lora.Linear(
                  (base_layer): Conv1D(nf=2304, nx=768)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=2304, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (c_proj): Conv1D(nf=768, nx=768)
                (attn_dropout): Dropout(p=0.1, inplace=False)
                (resid_dropout): Dropout(p=0.1, inplace=False)
              )
              (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (mlp): GPT2MLP(
                (c_fc): Conv1D(nf=3072, nx=768)
                (c_proj): Conv1D(nf=768, nx=3072)
                (act): NewGELUActivation()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (dropout): Dropout(p=0.1, inplace=False)
  )
)
```