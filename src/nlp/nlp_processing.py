import spacy
from spacy.matcher import Matcher

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Initialize matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# Add pattern for Train IDs (like 'T1000')
pattern = [{"TEXT": {"REGEX": "T\d+"}}]
matcher.add("TRAIN_ID", [pattern])

# Add pattern for Stations (assuming they follow a consistent naming convention like 'Station_A')
pattern = [{"TEXT": {"REGEX": "Station_[A-Z]"}}]
matcher.add("STATION", [pattern])

# Add pattern for Times (like '360 minutes')
pattern = [{"IS_DIGIT": True}, {"LOWER": "minutes"}]
matcher.add("TIME", [pattern])

# Add pattern for Delays (like '5 minutes delay')
pattern = [{"IS_DIGIT": True}, {"LOWER": "minutes"}, {"LOWER": "delay"}]
matcher.add("DELAY", [pattern])

def extract_schedule_info(text):
    """
    Extract schedule-related information from a given text.
    """
    # Process the text through the Spacy NLP pipeline
    doc = nlp(text)

    # Find matches in the processed text
    matches = matcher(doc)

    # Create a dictionary to hold the matched information
    schedule_info = {}

    # Iterate over the matches and update the dictionary
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # Get the rule's unique ID
        span = doc[start:end]  # Get the matched span
        schedule_info[rule_id] = span.text

    return schedule_info

# Example usage
if __name__ == "__main__":
    text = "Train T1000 arrives at Station_A at 360 minutes with a delay of 5 minutes delay."
    print(extract_schedule_info(text))
