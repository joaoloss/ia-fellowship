# Notes, Observations, Partial results, Conclusions, Curiosities

## Useful links

1. [OpenAI API Princing](https://platform.openai.com/docs/pricing?latest-pricing=standard)
2. [OpenAI Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
3. [OpenAI Guide for GPT-5](https://platform.openai.com/docs/guides/latest-model?quickstart-panels=fast)
4. [GPT-5-mini page](https://platform.openai.com/docs/models/gpt-5-mini)
5. [File inputs](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses)
6. [Structured model outputs](https://platform.openai.com/docs/guides/structured-outputs)
7. [Latency optimization](https://platform.openai.com/docs/guides/latency-optimization)
8. [OpenAI prompt guide](https://platform.openai.com/docs/guides/prompting)
9. [Response Object Specification](https://platform.openai.com/docs/api-reference/responses)

## Partial results

1. Adding `reasoning={"effort": "minimal"}` improved significantly: from ~20s to ~3s and from ~1600 total used tokens to ~320 total used tokens.
    - Obs.: the default is value is `medium`.
2. Sending the PDF file to LLM (via base64) instead of the text of the PDF in prompt: ~2.5x more time, ~2.4x more tokens and ~2x more cost.
3. Passing the PDF text in a matrix format (preserving the PDF layout) showed improved accuracy compared to using the raw text, although it slightly increased token usage.
4. Passing the `extraction_schema` as YAML instead of JSON helped reduce the number of input_tokens, but not significantly.

## About my Heuristic

### Limitations

1. The heuristic only looks to present values, not present aren't considered.