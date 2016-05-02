# dis3

`dis3` is a Python 2.7 backport of the `dis` module from Python 3.5.


## Documentation

See the [Python 3 `dis` module documentation][dis-py3].


## Caveats

### Strings

Python 2 and Python 3 handle strings differently. The `dis3` module
follows the Python 3 convention and interprets `str` as bytes. Use
`unicode` where the docs specify a "string"; you will not get the
expected result if you pass a `str` of source code.

### Keyword-only Parameters

Python 2 has no concept of keyword-only parameters, so those parameters
have been changed to standard keyword parameters. For instance:

- `dis.dis(x=None, file=None)`

### Bytecode Instructions

See the [Python 2 `dis` module documentation][dis-py2] for the list of
bytecode instructions actually used in Python 2.


## License

MIT / PSF. See `LICENSE`.


[dis-py2]: https://docs.python.org/2/library/dis.html
[dis-py3]: https://docs.python.org/3/library/dis.html
