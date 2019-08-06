from pytorch_transformers.convert_pytorch_checkpoint_to_tf import convert_pytorch_checkpoint_to_tf
from pytorch_pretrained_bert import BertModel

model = BertModel.from_pretrained("./finetuned_lm")
convert_pytorch_checkpoint_to_tf(model, "./finetuned_lm_tf", "fine_tuned_tf")