# NLPiper

[![Test](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml/badge.svg)](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

(Introduction)

---
## Install

You can install NLPiper from PyPi with `pip` or your favorite package manager:

    pip install nlpiper

---

## Optional Dependencies

Some **transformations** require the installation of additional packages.
The following table explains the optional dependencies that can be installed:

| Package                                                                                                   | Description
|---                                                                                                        |---
| <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" target="_blank"><code>bs4</code></a>     | Used in **CleanMarkup** to remove HTML and XML from the document.
| <a href="https://www.nltk.org/install.html" target="_blank"><code>nltk</code></a>                         | Used in **RemoveStopWords** to remove stop words from the document.
| <a href="https://github.com/alvations/sacremoses" target="_blank"><code>sacremoses</code></a>             | Used in **MosesTokenizer** to tokenize the document using Sacremoses.

To install the optional dependency needed for your purpose you can run:


    pip install nlpiper[<package>]


You can install all of these dependencies at once with:


    pip install nlpiper[all]


---

## Usage

---

## Development Installation

```
git clone https://github.com/tomassosorio/NLPiper.git
cd NLPiper
poetry install
```

To install an [optional dependency](##Optional-Dependencies) you can run:


    poetry install --extras <package>


To install all the optional dependencies run:


    poetry install --extras all


---

## Contributions

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](CONTRIBUTING.md)
on GitHub.

---

## Issues

Go [here](https://github.com/tomassosorio/NLPiper/issues) to submit feature
requests or bugfixes.

---

## License and Credits

`NLPiper` is licensed under the [MIT license](LICENSE) and is written and
maintained by Tomás Osório ([@tomassosorio](https://github.com/tomassosorio)), Daniel Ferrari ([@FerrariDG](https://github.com/FerrariDG)) and Carlos Alves ([@cmalves](https://github.com/cmalves))
