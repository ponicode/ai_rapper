import os
#import gpt_2_simple as gpt2

class Rapper:
    """Ponicode Rapper generates lyrics by training on rap songs"""

    def __init__(self):
        self.model_name = '124M'
        self.steps = 500
        self.run_name = 'rap'
        self.print_every = 10
        self.sample_every = 200
        self.save_every = 500

    def get_texts_attributes(self, lyrics_dir):
        lyrics_files_paths = [file_path for file_path in os.listdir(lyrics_dir) if '.txt' in file_path and 'lyrics' in file_path]
        texts_attributes = []
        for file_path in lyrics_files_paths:
            text_attributes = {}
            print(file_path)
            author_name = file_path.split('_')[0]
            with open(file_path, 'r') as f:
                text = f.read()
            text_attributes['author_name'] =  author_name
            text_attributes['text'] = text
            texts_attributes.append(text_attributes)
        return texts_attributes

    def process_texts(self, lyrics_dir):
        texts_attributes = self.get_texts_attributes(lyrics_dir)
        texts = ''
        for text_attribute in texts_attributes:
            text = text_attribute['text']
            text = '<|startoftext|>\n' + text_attribute['author_name'] + ' style\n' + text + '\n<|endoftext|>\n'
            texts += text
        self.all_lyrics_path = 'all_lyrics/all_lyrics.txt'
        if not os.path.exists('all_lyrics'):
            os.makedirs('all_lyrics')
        with open(self.all_lyrics_path, 'w') as f:
            f.write(texts)
        return texts

    def download_model(self):
        return gpt2.download_gpt2(model_name=self.model_name)

    def train(self, lyrics_paths):
        self.sess = gpt2.start_tf_sess()
        gpt2.finetune(self.sess,
                      dataset=lyrics_paths,
                      model_name=self.model_name,
                      steps=self.steps,
                      restore_from='fresh',
                      run_name=self.run_name,
                      print_every=self.print_every,
                      sample_every=self.sample_every,
                      save_every=self.save_every
                      )
    
    def generate_lyrics(self):
        return gpt2.generate(self.sess, self.run_name)