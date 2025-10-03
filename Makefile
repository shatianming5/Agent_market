PY?=python

.PHONY: setup dev test lint fmt run quickcheck clean

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -r server/requirements.txt
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q -rs

run:
	$(PY) -m uvicorn server.main:app --host 127.0.0.1 --port 8032

quickcheck:
	$(PY) scripts/server_quickcheck.py

dev-check:
	$(PY) scripts/start_both_and_test.py

clean:
	rm -f user_data/app.db
	rm -rf .pytest_cache
