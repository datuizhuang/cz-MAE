from AttrDict import AttrDict

config = AttrDict()
_C = config

# for task
_C.TASK = AttrDict()
task = _C.TASK
task.NAME = 'default task name'
task.LOG_NAME = 'default log name'

# for build model
_C.MODEL = AttrDict()
_C.MODEL.PATCH_SIZE = 16
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.DECODER_NUM_CLASSES = _C.MODEL.IN_CHANNELS * _C.MODEL.PATCH_SIZE ** 2
_C.MODEL.ENCODER_EMBED_DIM = 1024
_C.MODEL.DECODER_EMBED_DIM = 512
_C.MODEL.ENCODER_DEPTH = 8
_C.MODEL.DECODER_DEPTH = 6
_C.MODEL.ENCODER_HEADS = 8
_C.MODEL.DECODER_HEADS = 4
_C.MODEL.MLP_RATIO = 4
_C.MODEL.DROP_RATE = 0
_C.MODEL.ATTN_DROP_TATE = 0
_C.MODEL.DROP_PATH_RATE = 0
_C.MODEL.ENCODER_QKV_BIAS = True
_C.MODEL.DECODER_QKV_BIAS = False

# for dataset
_C.DATA = AttrDict()
_C.DATA.ROOT = ''
_C.DATA.DATAFILE = ''
_C.DATA.IMAGE_SIZE = (224, 224)
_C.DATA.CROP = 0.2
_C.DATA.AUG = 'NULL'
_C.DATA.BATCH_SIZE = 64
_C.DATA.NUM_WORKERS = 4

# for optim
_C.OPTIM = AttrDict()
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.LR_SCHEDULER = 'COSINE'
_C.OPTIM.OPTIM = 'sgd'
_C.OPTIM.WARM_EPOCH = 5
_C.OPTIM.WARM_MULTIPLIER = 100
_C.OPTIM.LR_DECAY_EPOCHS = [120, 160, 200]
_C.OPTIM.LR_DECAY_RATE = 0.1
_C.OPTIM.WEIGHT_DECAY = 1e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.AMP_OPT_LEVEL = 'O1'
_C.OPTIM.EPOCHS = 200
_C.OPTIM.START_EPOCH = 1

# for io
_C.IO = AttrDict()
_C.IO.PRINT_FREQ = 20
_C.IO.SAVE_FREQ=10
_C.IO.OUTPUT_DIR='./output'
_C.IO.POSTFIX = ''

# for misc
_C.MISC = AttrDict()
_C.MISC.RNG_SEED=0


if __name__ == '__main__':
    from AttrDict import merge_a_to_b
    print(config)

    dic = dict(TASK=dict(NAME='CHENZHUANG', LOG_NAME='chenzhuang'))
    merge_a_to_b(dic, config)
    print(config)
