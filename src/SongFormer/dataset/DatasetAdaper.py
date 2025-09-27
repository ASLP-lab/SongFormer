from abc import ABC, abstractmethod


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset adapter with necessary parameters.
        """
        raise NotImplementedError("Subclasses must implement the __init__ method.")

    @abstractmethod
    def get_ids(self):
        """
        Get the IDs of the dataset.
        This method should be implemented by subclasses.

        Returns:
            A list or set of IDs representing the dataset. In format: ID + start_time
            must cosider the split of dataset, e.g. train, val, test.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_item_json(self, *args, **kwargs):
        """
        Get the item JSON representation from the dataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")
