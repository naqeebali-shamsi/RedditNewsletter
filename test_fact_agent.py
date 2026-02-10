import sys
import io

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from execution.agents import FactVerificationAgent

test_article = '''
Meta's Llama 3 70B Instruct model reportedly achieves around 82% on the MMLU benchmark, placing it in the same ballpark as Claude 3.5 Sonnet on general knowledge tasks.
Some analysts claim the broader AI market will barely reach $500 billion by 2027, while others project it could approach $1 trillion depending on how you count AI-related hardware and services.
GPT-4 Turbo models are advertised with a 128,000-token context window, but many developers assume this also means they can reliably generate 128,000 tokens in a single response.
Early blog posts sometimes confuse Llama 3.1 70B with Llama 3.3 70B when reporting MMLU scores, leading to benchmark tables that mix results from different model families.
Industry reports frequently round AI market forecasts, turning ranges like $780-990 billion into simplified headlines such as "$800B AI market by 2027," which can hide underlying uncertainty.
'''

agent = FactVerificationAgent()
report = agent.verify_article(test_article, 'AI Models')
print(report.summary)
print(f'Passes quality gate: {report.passes_quality_gate}')
