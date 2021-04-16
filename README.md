# NLPiper

[![Test](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml/badge.svg)](https://github.com/tomassosorio/NLPiper/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

(Introduction)

---
## Install

The package can be install using `pip`:

---

## Usage

---

## Development Installation

```
git clone https://github.com/tomassosorio/NLPiper.git
cd NLPiper
poetry install
```

**NLPiper** by design, will only install the minimum necessary dependencies, however some **transformations** need special
packages, to install them you can install them all at once by running:

```
poetry install --extra 'all'
```

Or only choose the ones that are strictly needed for your purpose, replacing `'all'` by the ones needed. 

Available options:
- `'bs4'`
- `'nltk'`
- `'sacremoses'`

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
