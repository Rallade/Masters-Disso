from data_cleanup import cleanup
from duplicate_remove import remove_dupes
import csv


filename = "cotswolds.csv"


cleanup(filename, "clean_data.csv")
remove_dupes("clean_data.csv", "no_duplicates.csv")
print("File cleanup complete")

"""
DB upload
"""

from db_helpers import DB_helpers
db = DB_helpers("cotswoldsdata")
#REMOVE LINE WHEN DONE
db.drop()

db.upload_csv("no_duplicates.csv")

print("Database upload complete")

"""
Fine Tuning
"""
import fine_tune.main

#Very RAM intensive, consider changing parameters, lowering batch size (by divisions of 2) and increasing gradient steps (by multiples of 2)
#Accuracy trade off
#fine_tune.main.tune(db, epochs=3, train_batch_size=32, gradient_accumulation_steps=1, fraction_used=0.1)

"""
Launch BERT

"""
batch_size = 256

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
args = get_args_parser().parse_args(['-model_dir', '../multi_cased_L-12_H-768_A-12/',
                                     '-tuned_model_dir', 'fine_tune/finetuned_full_lm_tf',
                                     '-ckpt_name', 'fine_tuned_tf.ckpt',
                                     '-pooling_strategy', "NONE",
                                     '-max_seq_len', 'NONE',
                                     '-num_worker', '1',
                                     '-show_tokens_to_client',
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     ])
server1 = BertServer(args)
server1.start()


"""
Encoding
"""
from base_embeddings import create
create(db, number_of_processors=1, batch_size=batch_size) #No of GPU (or CPUs as a fallback)

#band-aid solution
import fix_missing
fix_missing.fix(db)

"""
Pooling
"""

from add_pooling_to_DB import create_basic_embeddings_appended_title, create_nltk_pos_embeddings_appended_title
create_basic_embeddings_appended_title(db)
create_nltk_pos_embeddings_appended_title(db)