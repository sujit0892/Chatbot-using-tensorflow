import numpy as np
import tensorflow as tf

class MemoryNetwork(object):
    def __init__(self, sentence_size, vocab_size, candidates_size, 
                 candidates_vec, embedding_size, hops, 
                 initializer=tf.random_normal_initializer(stddev=0.1), 
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                 session=tf.Session()):
        self._hops = hops
        self._candidates_vec = candidates_vec
        
        self._facts = tf.placeholder(
            tf.int32, [None, None, sentence_size], name="facts")
        self._questions = tf.placeholder(
            tf.int32, [None, sentence_size], name="questions")
        self._answers = tf.placeholder(
            tf.int32, [None], name="answers") 

        with tf.variable_scope("MemoryNetwork"):
            self.word_emb_matrix = tf.Variable(initializer(
                [vocab_size, embedding_size]), name="A")
            self.transformation_matrix = tf.Variable(initializer(
                [embedding_size, embedding_size]), name="H")
            self.output_word_emb_matrix = tf.Variable(initializer(
                [vocab_size, embedding_size]), name="W")

        logits = self._inference(self._facts, self._questions)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(
            cross_entropy, name="cross_entropy_sum")


        self.loss_op = cross_entropy_sum


        grads_and_vars = optimizer.compute_gradients(self.loss_op)
        

        self.train_op = optimizer.apply_gradients(
            grads_and_vars, name="train_op")

        self.predict_op = tf.argmax(logits, 1, name="predict_op")


        self._session = session
        self._session.run(tf.initialize_all_variables())
        
    def _input_module(self, facts):
        with tf.variable_scope("InputModule"):
            facts_emb = tf.nn.embedding_lookup(self.word_emb_matrix, 
                                               facts)
            return tf.reduce_sum(facts_emb, 2)
    
    def _question_module(self, questions):
        with tf.variable_scope("QuestionModule"):
            questions_emb = tf.nn.embedding_lookup(
                self.word_emb_matrix, questions)
            return tf.reduce_sum(questions_emb, 1)
        
    def _memory_module(self, questions_emb, facts_emb):
        with tf.variable_scope("MemoryModule"):
            initial_context_vector = questions_emb
            context_vectors = [initial_context_vector]
            for hop in range(self._hops):
                context_temp = tf.transpose(
                    tf.expand_dims(context_vectors[-1], -1), [0, 2, 1])
                similarity_scores = tf.reduce_sum(
                    facts_emb * context_temp, 2)
                probs = tf.nn.softmax(similarity_scores)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), 
                                          [0, 2, 1])
                facts_temp = tf.transpose(facts_emb, [0, 2, 1])
                context_rep = tf.reduce_sum(facts_temp*probs_temp, 2)
                context_vector = tf.matmul(context_vectors[-1], 
                                           self.transformation_matrix) \
                                 + context_rep
                context_vectors.append(context_vector)
            return context_vector
        
    def _output_module(self, context_vector):
        with tf.variable_scope("OuptutModule"):
            candidates_emb = tf.nn.embedding_lookup(self.output_word_emb_matrix, 
                                                    self._candidates_vec)
            candidates_emb_sum = tf.reduce_sum(candidates_emb, 1)
            return tf.matmul(context_vector, tf.transpose(candidates_emb_sum))
    
    def _inference(self, facts, questions):
        with tf.variable_scope("MemoryNetwork"):
            input_vectors = self._input_module(facts)
            question_vectors = self._question_module(questions)
            context_vectors = self._memory_module(question_vectors, 
                                                  input_vectors)
            output = self._output_module(context_vectors)
            return output
    
    def fit(self, facts, questions, answers):
        feed_dict = {self._facts: facts, 
                     self._questions: questions, 
                     self._answers: answers}
        loss, _ = self._session.run([self.loss_op, self.train_op], 
                                    feed_dict=feed_dict)
        return loss

    def predict(self, facts, questions):
        feed_dict = {self._facts: facts, self._questions: questions}
        return self._session.run(self.predict_op, feed_dict=feed_dict)
