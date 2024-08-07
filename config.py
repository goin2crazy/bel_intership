class Config():

  mainpage_url = "https://injapan.ru/category/2084017018/currency-USD/mode-1/condition-used/page-1/sort-enddate/order-ascending.html"
  model_path = 'checkpoint.weights.h5'

  image_size = (512, 512)
  image_channels = 3
  image_shape = (*image_size, image_channels)

  batch_size = 32

class Logs():
  runtimes = ''

  def __call__(self, log_text):
    print(log_text)
    self.runtimes += f"\n{log_text}"
    return self.runtimes

  def pop(self):

    runtimes_text = str(self.runtimes)
    self.runtimes = ''
    return runtimes_text

cfg = Config
logs = Logs()