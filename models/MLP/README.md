## Configuration structure

- SYSTEM
    - USE_GPU (bool) [optional]: whether to use GPU for training and evaluating
    - DEVICE_ID (int) [optional]
- MODEL
    - NAME (str) [**required**]: model name
    - DIR (str) [**required**]: path from the content root
    - SAVE_NAME (str) [optional]: save some key information in the name
    - DATE (str) [hidden]: model running time (automatically added to the parameters)
- TRAIN
    - NUM_EPOCHS (int) [optional]: number of epochs
    - PRETRAIN (bool) [optional]: whether to load the pretrained model
    - PRETRAIN_DIR (str) [optional]: pretrained model save directory. **If PRETRAIN is true, this parameter must be specified.**
    - NUM_USERS (int) [hidden]: users number (automatically added to the parameters)
    - NUM_ITEMS (int) [hidden]: items number (automatically added to the parameters)
    - DENSITY (float) [hidden]: density used in current training (automatically added to the parameters)
- TEST
