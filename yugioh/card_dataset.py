import csv
from collections import UserList
from pathlib import Path
from yugioh.card import Card, attr_mapping

class CardDataset(UserList):
    def __init__(self, path):
        super().__init__()
        if isinstance(path, str):
            path = Path(path)

        self.card_index = dict()
        self.index_card = dict()
        self.variants = dict()

        with open(path / "cards.csv", encoding="utf8") as fp:
            reader = csv.DictReader(fp)
            for idx, card in enumerate(reader):
                self.data.append(Card(**{attr_mapping.get(attr, attr): val for attr, val in card.items()}))
                self.card_index[card["id"]] = idx
                self.index_card[idx] = card["id"]

        with open(path / "variants.csv", encoding="utf8") as fp:
            reader = csv.reader(fp)
            next(reader)
            for row in reader:
                self.variants[row[0]] = row[1]

    def resolve_to_index(self, card_id):
        return self.card_index[self.variants.get(card_id, card_id)]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, str):
            return self.data[self.card_index[key]]
        raise NotImplemented

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.index_to_card
        if isinstance(key, str):
            return key in self.card_index
        raise NotImplemented
