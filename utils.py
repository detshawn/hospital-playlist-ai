from PIL import Image
from IPython.display import display
from tensorboardX import SummaryWriter

""" Util Functions """


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    display(img)
    img.save(filename)


def post_process_image(data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    # img = Image.fromarray(img)

    return img


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)
        self.writer.flush()

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)
        self.writer.flush()

    def graph_summary(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model=input_to_model, profile_with_cuda=torch.cuda.is_available())
        self.writer.flush()

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)
        self.writer.flush()

    def image_summary(self, tag, value, step, dataformats='CHW'):
        self.writer.add_image(tag, value, step, dataformats=dataformats)
        self.writer.flush()
