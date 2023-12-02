# super-easy-super-resolution
- 😊 a super easy and small project to realise super resolution (real-esr)
- ☕ just clone & pip install some libs in common use, then you can run ! 

## 👌 Super easy install

```bash
$ git clone https://github.com/LSTM-Kirigaya/super-easy-super-resolution
$ pip install opencv-python numpy pillow torch colorama tqdm
```

## 🙌 Super easy usage

```bash
$ python real-esrgan.py -i image/test.jpg -o test.sr.jpg
```

## 🐳 Use SR in your browser

If you don't like command usage, we support use SR in a web app:

First install streamlit
```bash
$ pip install streamlit
```

Then, run streamlit:
```bash
$ python -m streamlit run app.py
```

The command will automatically open a web page in your browser:

<center>
<img src="./image/streamlit.png" alt="streamlit" style="width: 80%; height: auto;">
</center>

---

## Appendix: Compare reconstruction quality

|   test.jpg    |   test.sr.jpg    |
|:------------:|:------------:|
|  <img src="./image/test.jpg" alt="Image 1" style="width: 600px; height: auto;">  |  <img src="./test.sr.jpg" alt="Image 2" style="width: 600px; height: auto;">  |