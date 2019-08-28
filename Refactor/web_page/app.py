from flask import Flask, render_template, make_response, request, after_this_request, g
import sys
import random
sys.path.insert(1, '../')
from search import Search
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
args = get_args_parser().parse_args(['-model_dir', '../../multi_cased_L-12_H-768_A-12/',
                                     '-tuned_model_dir', '../fine_tune/finetuned_full_lm_tf',
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

search_mean = Search("cotswaldsdata", "mean_pooling")
search_max = Search("cotswaldsdata", "max_pooling_total")
search_mean_pos = Search("cotswaldsdata", "mean_pooling_pos_filtered")

app = Flask(__name__)


@app.before_request
def set_seed():
    seed = request.cookies.get('seed')
    if seed is None:
        seed = random.randint(1,6)
        @after_this_request
        def save(response):
            response.set_cookie('seed', str(seed))
            return response
        g.seed = seed


@app.route('/')
def index():
    try:
        seed = g.seed
    except AttributeError:
        seed = request.cookies.get('seed')

    return render_template('index.html', order=seed)

@app.route('/search/<pooling>/<user_query>')
def search(user_query, pooling):
    if pooling == "mean_pooling":
        results = search_mean.query(user_query)
        return render_template('search_results.html', results=results)
    elif pooling == "max_pooling_total":
        results = search_max.query(user_query)
        return render_template('search_results.html', results=results)
    elif pooling == "mean_pooling_pos":
        results = search_mean_pos.query(user_query)
        return render_template('search_results.html', results=results)


if __name__ == "__main__":
    app.run(debug=False)