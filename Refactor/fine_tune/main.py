from .generate_data import generate
from .pregenerate_training_data import main as ptd_main
from .finetune_on_pregenerated import main as fop_main
from pytorch_transformers.convert_pytorch_checkpoint_to_tf import convert_pytorch_checkpoint_to_tf
from pytorch_pretrained_bert import BertModel
import sys


def tune(db, epochs=3, train_batch_size=32, gradient_accumulation_steps=1, fraction_used=0.5):
    """
    Create fine-tuned model
    sys args are used since the original scripts
    were meant to be run from a CLI
    """
    generate(db.find(), fraction_used)
    args="pregenerate_training_data --train_corpus training_data_titles_reviews.txt --bert_model bert-base-multilingual-cased --output_dir training_full/ --epochs_to_generate " + \
        str(epochs) + " --max_seq_len 256"
    sys.argv = args.split()
    ptd_main()
    args = "finetune_on_pregenerated.py --pregenerated_data training_full/ --bert_model bert-base-multilingual-cased --output_dir finetuned_full_lm/ --epochs " + \
        str(epochs) + " --train_batch_size " + str(train_batch_size) + \
        " --gradient_accumulation_steps " + str(gradient_accumulation_steps)
    sys.argv = args.split()
    print(sys.argv)
    fop_main()
    model = BertModel.from_pretrained("./finetuned_full_lm")
    convert_pytorch_checkpoint_to_tf(model, "./finetuned_full_lm_tf", "fine_tuned_tf")
    
