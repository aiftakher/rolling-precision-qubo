# Install

Editable development install:

```bash
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

Optional D-Wave hardware/hybrid support:

```bash
python -m pip install -e ".[dwave]"
```

Normal package use does not require D-Wave Leap credentials. If hardware is
used, configure credentials through the standard D-Wave environment variables
or D-Wave config file. Never commit API tokens.
