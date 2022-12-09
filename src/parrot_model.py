"""Module for a Parrot Model class"""

import torch
from parrot import Parrot


class ParrotModel:
    """A class for the parrot model"""

    def __init__(self, seed: int = 0) -> None:
        """Creates parrot"""
        self.parrot = Parrot(
            model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False
        )
        self.__seed = seed

    def set_seed(self, seed: int) -> None:
        """Sets seed for parrot and updates it in cuda and torch"""
        self.__seed = seed
        self.__set_seed()

    def __set_seed(self) -> None:
        """Sets seed for torch"""
        torch.manual_seed(self.__seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.__seed)

    def para_phrase(
        self,
        phrase: str,
        do_diverse: bool = True,
        amount: int = 1,
        fluency_threshold: int = 0.8,
        adequacy_threshold: int = 0.8,
    ) -> list:
        """Get the list of paraphrases

        Arguments:
        do_diverse -- do diverse
        amount -- maximum length that will be returned
        fluency_threshold -- fluency threshold (Is the paraphrase fluent English?)
        adequacy_threshold -- adequacy threshold (Is the meaning preserved adequately?)

        Returns:
        the list of generated paraphrases, or None if nothing was generated
        """
        para_phrases = self.parrot.augment(
            input_phrase=phrase,
            diversity_ranker="levenshtein",
            do_diverse=do_diverse,
            max_return_phrases=amount,
            fluency_threshold=fluency_threshold,
            adequacy_threshold=adequacy_threshold,
        )

        return para_phrases
