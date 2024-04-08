use std::marker::PhantomData;

pub struct RecSplit<T> {
    _phantom: PhantomData<T>,
}

impl<T> RecSplit<T> {
    pub fn new(values: &[T]) -> Self {
        todo!()
    }

    pub fn hash(&self, value: &T) -> usize {
        todo!()
    }

    pub fn serialize(&self) -> Box<[u8]> {
        todo!()
    }

    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
