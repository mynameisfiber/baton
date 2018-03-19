This is a simple example of federated learning which learns coefficients of a linear model
in decentralized fashion.

## Run example (locally)

Add project root to PYTHONPATH:
```bash
export PYTHONPATH=<project-root-dir>
```

Start a manager process:
```bash
python demo.py --role manager --port 8080
```

Start one or more worker processes:
```bash
python demo.py --role worker --manager-host localhost --manager-port 8080 --port 8081
```

Run a round of training:
```bash
curl -X GET localhost:8080/lineartest/start_round
```