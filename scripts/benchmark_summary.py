"""Compare a local OpenAI-compatible model; outputs remain under ignored data/."""
import argparse
import json
import time
from pathlib import Path

import requests

from ai_transcriber.llm.prompts import MEETING_SUMMARY_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, default=Path('data/summary-benchmark'))
    args = parser.parse_args()
    cases = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/summary_cases.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        start = time.monotonic()
        response = requests.post(args.base_url.rstrip('/') + '/chat/completions', json={
            'model': args.model, 'temperature': .2, 'seed': 42, 'max_tokens': 1400,
            'messages': [{'role': 'system', 'content': 'Ты анализируешь рабочие встречи. Отвечай на русском языке.'},
                         {'role': 'user', 'content': MEETING_SUMMARY_PROMPT.format(text=case['text'])}],
        }, timeout=420)
        response.raise_for_status()
        data = response.json()
        text = data['choices'][0]['message'].get('content') or ''
        (args.output / (case['id'] + '.md')).write_text(text, encoding='utf-8')
        records.append({'case': case['id'], 'seconds': round(time.monotonic() - start, 2),
                        'finish_reason': data['choices'][0]['finish_reason'],
                        'usage': data.get('usage'), 'timings': data.get('timings'),
                        'fact_mentions': {fact: fact.casefold() in text.casefold() for fact in case['facts']},
                        'manual_review_required': case['checks']})
        print(case['id'], records[-1]['seconds'], 'seconds')
    (args.output / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Fact mentions are a smoke check; manually review negation, attribution and omitted facts.')


if __name__ == '__main__':
    main()
