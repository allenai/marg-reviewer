## MARG

This is the repository for [MARG: Multi-Agent Review Generation for Scientific Papers](https://arxiv.org/pdf/2401.04259).  It contains the code for the web interface that was used for the user study in the paper, and functions as a demo of our system.

To run the demo, install [Docker](https://www.docker.com/get-started) (and potentially docker-compose, depending on whether your Docker version has `docker compose` built in).  You will also need to create a file called `.env` with
```
OPENAI_API_KEY=your_key
AWS_SECRET_ACCESS_KEY=your_key
AWS_ACCESS_KEY_ID=your_key
```

At a minimum, the `OPENAI_API_KEY` needs to be a valid API key.  However, the AWS keys are optional, and are only necessary for sending notification emails with Amazon SES (see below).  If you want to support email sending, you also need to modify `review_worker/run_reviewgen.py` and change the `OUTGOING_EMAIL` near the top of the file to the email address you configured with SES.

When the prerequisites are set up, simply run
```
sudo docker compose up --build
```

The web-based demo should then be running on localhost at port 8080.

When you submit a paper on the main page, reviews will be generated using SARG-B (referred to as "barebones" in the code), LiZCa ("liang_etal"), and MARG-S ("multi_agent_specialized").  If you set up email sending, you will receive an email notification when the reviews are finished generating; otherwise, you will need to check either the console output or the http://localhost:8080/list-results page to see when reviews are done and get the link to results.

Note that the result page URL may start with `/survey/` (from the email) or `/result/` (from the list-results page).  The "result" page is best for local use; the survey page hides method names and randomizes review order.


## Reproducing paper experiments

The code for alignment/metrics is in the `review_worker/paper_align_eval_repro.py` file, and configs for the experiments from Table 2 of the paper are in `review_worker/data/paper_align_eval/`, along with outputs (generated reviews, alignments, and logs).  A cache file for the gpt requests is provided in `review_worker/data/gpt3_cache.sqlite.xz`.  To run the full experiments:

1. Decompress gpt cache: `unxz review_worker/data/gpt3_cache.sqlite.xz`
2. Download the [aries dataset](https://github.com/allenai/aries), which has the paper texts and human reviewer data: `aws s3 sync --no-sign-request s3://ai2-s2-research-public/aries/ review_worker/data/aries/ && tar -C review_worker/data/aries/ -xf review_worker/data/aries/s2orc.tar.gz`
3. Enter the `review_worker` container: `docker build -t marg review_worker && docker run -it --rm --entrypoint /bin/bash -v $(realpath review_worker/data/):/reviewgen/data marg`
4. Run experiments: `for d in data/paper_align_eval/*; do python paper_align_eval_repro.py $d/align_config.json; done`
5. Results are in the corresponding `output/` dir, e.g. `data/paper_align_eval/marg_s/output/`

By default, following those steps will use the gpt cache from the paper experiments to ensure reproducibility; it can be disabled by deleting the `review_worker/data/gpt3_cache.sqlite` file or modifying `align_config.json` to not point to it.

## License

Copyright 2023 The Allen Institute for Artificial Intelligence

The code in this repository is licensed under Apache 2.0 (see the LICENSE file).
