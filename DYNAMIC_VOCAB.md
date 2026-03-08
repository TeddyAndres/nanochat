20260308 Dynamic Vocab implementation on nanochat baseline model

The Concept:

Dynamic vocabulary sparsity (dynamic vocab, sparse) is a novel concept. It does not exist in the research anywhere today and so all considerations will need to be evaluated from first principles. Dynamic vocab relates to training large language models (llms)

The principle novel idea with dynamic vocab is any model tables that have a vocabulary dimension (e.g. embedding dims, value embeds, lm_head) never get loaded as a full table onto the GPU. The master tables for these are stored on CPU and system RAM. Each micro batch of training (e.g. 2k sequence length x 32 batches = 64k tokens) a CPU process runs to evaluate unique tokens in the next training data set. A CPU process then gathers the data tables that have vocabulary length only for the rows matching the unique tokens in the training data for the upcoming microbatch. The CPU then passes the uniques tables in one go to the GPU before the start of the forward pass. The GPU then calculates the forward and backward pass on the training data and using the uniques tables (U) rather than the full vocabulary (V). After the optimizer step the GPU passes back the updated weight for U to the CPU to update on the master tables. The next micro batch of U and training data is passed from CPU to the GPU and the process begins again.

Some points:
- tables with V dimension never get passed to the GPU
- GPU VRAM saving because we do not need to load the full vocabulary ever
- GPU operations saving because matmul and other calculations are done on U not V
- no need for cold rows (token rows that are not in the current training set) because we always only have U for hot rows
- softmax only over hot rows
- no need for adam W decay because we do not have cold rows in the model. When new hot rows are added they get full gradients.


Optimizations:
- minimize CPU to GPU handoff time
    - use data loader to:
        - identify U for next training step
        - create mapping between each token in U and its position in the various tables (this will be relevant at the end of the backward pass)
        - pass U dimension tables to the GPU (be ready by the time GPU is finished with previous work) (async)
        - while GPU is working evaluate current step U vs next step U
        - create 3 lists
            - tokens in U that are in both current step and next step
            - tokens in current step U that are not in next step U
            - tokens in next step U that are no in current step U
        - when GPU is finished optimizer step pass (async) list of tokens that are in current step U but not in next step U to GPU with table locations
        - after GPU passes updated weights for tokens that are in current step but not in next step update (async) master tables with updated weights
    - use GPU to:
        - received updated U tables
        - do forward and backward passes
        - after receiving list of tokens in current step but not in next step pass table data to CPU
        - after receiving table data for tokens in next step but not in current step add data to existing table with tokens that were common between current step and next step
- minimize GPU VRAM
    - do not create extra cache tables
    - do not load full V dimension tables to VRAM
    - pass completed data back to CPU before loading new data
    - release VRAM where possible

Logging:
- we will need to log U per step


The above outline is a simplified representation of the implementation. The actual implementation should allow:
- an arbitrarily large vocab to run in VRAM because the tables are dimensioned in U rather than V
- no operations in the model calcs should reference V; the model only needs to know about U
- the changes should be fully implemented with the current model structure including hyperparameters, logging
    - do not need to keep backward compatibility
- be surgical in changes
    - change only the necessary aspects and do not make sweeping changes
