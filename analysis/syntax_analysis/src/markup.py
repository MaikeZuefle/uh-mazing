import re


def tokenize_and_flag(text):
    """
    Split annotated disfluent text into (clean_word, is_disfluent) pairs.

    Handles:
      _word_      -> (word, True)
      _word_,     -> (word, True)  then  (',', False)  [trailing punct detached]
      _,_         -> (',', True)
      _, _        -> (',', True)   [split annotation artifact, underscore stripped]
      plain word  -> (word, False)
      lone _      -> skipped
    """
    # Detach trailing punctuation from closing underscore: _word_, -> _word_ ,
    text = re.sub(r'_([^_\s]+)_([,.])', r'_\1_ \2', text)
    result = []
    for token in text.split():
        if token.startswith('_') and token.endswith('_') and len(token) > 1:
            word = token[1:-1]
            if word:
                result.append((word, True))
        elif token == '_':
            pass  # artifact from _, _ split
        elif '_' in token:
            # Partial annotation: strip underscores and treat as disfluent
            word = token.replace('_', '')
            if word:
                result.append((word, True))
        else:
            result.append((token, False))
    return result
