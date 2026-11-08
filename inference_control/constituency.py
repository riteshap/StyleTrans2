from nltk.tree import Tree
import stanza
import platform
from stanza.models.common.doc import Sentence


class Constituency:
    def __init__(self):
        self.nlp = self._load_stanza()
        self.function_pos = self._load_function_pos()
        pass

    def _load_stanza(self):
        model_dir = r"F:\python_code\pre_model\stanza-en"
        if platform.system() == "Windows":
            model_dir = r"F:\python_code\pre_model\stanza-en"

        elif platform.system() == "Linux":
            model_dir = r"/root/private_data/stanza-en"

        nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,constituency",
            model_dir=model_dir,
            download_method=None
            )
        return nlp

    def _load_function_pos(self):
        FUNCTION_POS = {
            "DT", "PDT",
            "IN", "CC", "TO",
            "PRP", "PRP$", "WP", "WP$", "WDT",
            "POS", "EX",
            ".", ",", ":", "``", "''",
            "-LRB-", "-RRB-"
        }
        return FUNCTION_POS

    # return: sentence, tree
    def parse_sentence(self, text):
        """
        Stanza constituency parsing。
        return:
            stanza_sentence
            nltk_tree
        """
        doc = self.nlp(text)
        if len(doc.sentences) != 1:
            raise ValueError(f"Expected one sentence, but got {len(doc.sentences)} sentences.")
        sentence = doc.sentences[0]
        # Stanza ParseTree -> NLTK Tree
        tree = Tree.fromstring(str(sentence.constituency))

        return sentence, tree
    # return spans
    def extract_constituent_spans(self, tree, sentence, original_text):
        """
        Extract constituency spans.

        Constituency leaves are aligned with Stanza sentence.words,
        while character offsets are recovered from their parent tokens.

        Spans that cut through a multi-word token / contraction
        are discarded.
        """

        spans = []

        words = sentence.words
        tokens = sentence.tokens
        leaves = tree.leaves()

        # 1. Constituency leaves should align with WORDS,
        #    not surface TOKENS.
        if len(leaves) != len(words):
            raise ValueError(
                f"Tree-word mismatch: "
                f"{len(leaves)} leaves vs {len(words)} words.\n"
                f"Leaves: {leaves}\n"
                f"Words: {[w.text for w in words]}"
            )

        # 2. Map each syntactic word to its surface token
        word_to_token = {}

        for token_index, token in enumerate(tokens):
            for word in token.words:
                word_to_token[word.id - 1] = token_index

        # 3. Recursive traversal
        def walk(node, cursor, depth=0):

            if not isinstance(node, Tree):
                return cursor + 1

            start_word = cursor

            for child in node:
                cursor = walk(child, cursor, depth + 1)
            end_word = cursor  # right-open
            # Which surface token contains the first/last word?
            start_token = word_to_token[start_word]
            end_token = word_to_token[end_word - 1]

            first_token = tokens[start_token]
            last_token = tokens[end_token]

            # 4. Check whether span boundaries cut through
            #    a contraction / multi-word token.
            # First word of this surface token
            first_word_id = first_token.words[0].id - 1

            # Last word of this surface token
            last_word_id = last_token.words[-1].id - 1

            boundary_aligned = (
                    start_word == first_word_id
                    and (end_word - 1) == last_word_id
            )

            if boundary_aligned:
                start_char = first_token.start_char
                end_char = last_token.end_char

                span_text = original_text[start_char:end_char]

                pos_tags = [pos for _, pos in node.pos()]

                spans.append({
                    "label": node.label(),
                    # syntactic-word positions
                    "start_word": start_word,
                    "end_word": end_word,
                    # surface-token positions
                    "start_token": start_token,
                    "end_token": end_token + 1,
                    "start_char": start_char,
                    "end_char": end_char,
                    "word_count": end_word - start_word,
                    "words": [word.text for word in words[start_word:end_word]],
                    "pos_tags": pos_tags,
                    "text": span_text,
                    "depth": depth
                })

            return end_word

        walk(tree, cursor=0, depth=0)

        return spans

    # return results
    def filter_constituents(self, spans, max_words=4):

        CONTENT_POS = {
            "NN", "NNS",
            "JJ", "JJR", "JJS",
            "RB", "RBR", "RBS",
            "VB", "VBD", "VBG",
            "VBN", "VBP", "VBZ"
        }

        PROPER_NOUN_POS = {
            "NNP", "NNPS"
        }
        filtered = []
        for span in spans:
            pos_tags = set(span["pos_tags"])
            # 1. Length constraint
            if span["word_count"] > max_words:
                continue
            # 2. Cannot contain first token
            if span["start_token"] == 0:
                continue
            # 3. Remove proper nouns
            if pos_tags & PROPER_NOUN_POS:
                continue
            # 4. Must contain at least one editable content word
            if not (pos_tags & CONTENT_POS):
                continue

            filtered.append(span)

        best_by_position = {}
        for span in filtered:
            key = (span["start_token"], span["end_token"])
            if key not in best_by_position:
                best_by_position[key] = span
            else:
                old = best_by_position[key]

                if span["depth"] < old["depth"]:
                    best_by_position[key] = span

        results = list(best_by_position.values())


        results.sort(key=lambda x: (x["start_token"], x["end_token"]))

        return results

    def split_by_constituent(self, original_text, span):

        start = span["start_char"]
        end = span["end_char"]

        left = original_text[:start]
        blank = original_text[start:end]
        right = original_text[end:]

        masked_text = left + "[BLANK]" + right

        return {"left": left, "blank": blank, "right": right,
                "original_text": original_text,
                "masked_text": masked_text,
        }

    def get_mask_candidates(self, sentence_text, max_words=4):

        sentence, tree = self.parse_sentence(sentence_text)

        spans = self.extract_constituent_spans(tree, sentence, sentence_text)

        spans = self.filter_constituents(spans, max_words=max_words)

        results = []

        for span in spans:
            split_result = self.split_by_constituent(sentence_text, span)

            results.append({**span, **split_result})

        return results


if __name__ == "__main__":

    text = "The food was extremely good."
    st = Constituency()

    candidates = st.get_mask_candidates(text, max_words=4)

    for item in candidates:
        print(item)
        print()


