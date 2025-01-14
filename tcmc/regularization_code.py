# contains regularization code
# from commit:  MarioStanke on 13 Oct 2019

class TCMCProbability(tf.keras.layers.Layer):
    def __init__(self, model_shape, tree, num_leaves, gamma = 0., **kwargs):
        self.model_shape = model_shape
        self.M = np.prod(self.model_shape)
        self.tree = tree
        self.num_leaves = num_leaves
        self.gamma = gamma # regularization weight
        super(TCMCProbability, self).__init__(**kwargs)

    def build(self, input_shape):
        self.s = input_shape[-1]
        s = self.s
        M = self.M # np.prod(self.model_shape)
        
        # The parameters that we want to learn
        self.alphabet_size = s
        self.rates = self.add_weight(shape = (M, int(s*(s-1)/2)), name = "R", dtype = tf.float64,
                                     initializer = Dirichlet)
        
        # we use the inverse of stereographic projection to get a probability vector
        kernel_init = tf.initializers.constant(1.0 / (np.sqrt(s) - 1)) # this initializes pi with uniform distribution
        self.pi_inv = self.add_weight(shape=(M, s-1), name = "pi_inv", dtype = tf.float64,
                                      initializer = kernel_init)
        
        # scaling: model specific mutation rates
        self.rho = self.add_weight(shape = M, name = "rho", dtype = tf.float64, initializer = tf.initializers.constant(1.0))
        
        # Retrieve the row and column indices for
        # triangle matrix above the diagonal 
        mat_ind = np.stack(np.triu_indices(s, 1), axis = -1)
        iupper = tensor_utils.broadcast_matrix_indices_to_tensor_indices(mat_ind, (M, s, s)).reshape((M, -1, 3))
        self.iupper = tf.convert_to_tensor(iupper)
    @tf.function
    def rate_matrices_from_rates(self, rates, pi ):
        """ construct matrices from the rates, pi"""
        s = self.s
        iupper = self.iupper
        M = self.M
        with tf.name_scope("embed_rates"):
            Q = tf.scatter_nd(iupper, rates, shape=(M, s, s), name = "rate_matrix")
        with tf.name_scope("symmetrize"):
            Q = Q + tf.transpose(Q,(0, 2, 1), name = "transpose")
        with tf.name_scope("apply_stationary_probabilites"):
            Q = tf.multiply(Q, pi[:, None, :])
        with tf.name_scope("calculate_diagonals"):
            new_diagonal = tf.math.reduce_sum(-Q, axis = 2, name = "new_diagonal")
            Q = tf.linalg.set_diag(Q, new_diagonal, name = "apply_diagonal")
        with tf.name_scope("normalize_to_one_expected_mutation"):
            emut = -tf.reduce_sum(tf.multiply(pi, new_diagonal),
                                  axis = 1, name = "expected_mutations")
            # prevent division by 0
            emut = tf.maximum(emut, 1e-9)
        Q = tf.multiply(Q, 1.0 / emut[:, None, None])
        return Q

    def regularization_loss(self, Q):
        """ 
            This penalizes the entries of Q after normalize_to_one_expected_mutation
            The sum of square roots of off-diagonal entries encourages rather sparse
            rates (in a soft sense).
        """
        with tf.name_scope("regularization"):
            Qoffdiag = tf.linalg.set_diag(Q, diagonal = np.zeros([self.M, self.s]))
            Q2 = tf.sqrt(tf.maximum(Qoffdiag, 0)) # penalizes more uniform rates within rows
            reg_loss = self.gamma * tf.math.reduce_sum(Q2)
        return reg_loss

    #@tf.function
    def call(self, inputs, training = None):
        #(input_signature=(tf.TensorSpec(shape=[None,None,None], dtype=tf.float64),))
        # define local variable names
        rates = self.rates ** 2
        pi_inv = self.pi_inv
        rho = self.rho
        T = self.tree
        s = self.pi_inv.shape[-1] + 1
        M = self.pi_inv.shape[0]
        n = self.num_leaves
        k = self.tree.shape[-1] - n 
        X = inputs
        
        # map `pi_inv` to a probability vector: stationary_propabilities
        pi = inv_stereographic_projection(pi_inv) ** 2 # pi sums up to 1
        iupper = self.iupper
        # construct the transition rate matrices
        with tf.name_scope("Q"):
            Q = self.rate_matrices_from_rates(rates, pi)
            scaledQ = tf.multiply(Q, rho[:, None, None])

        A = []

        edges_start, edges_target = np.nonzero(T)
        for a in range(n, n+k):
            with tf.name_scope(f"A_{a}_calculation"):
                e_s = edges_start[edges_target == a]
                e_t = edges_target[edges_target == a]
                t = T[e_s, e_t]
                with tf.name_scope(f"P_{a}"):
                    #P_a = tf.linalg.expm(t[:,None,None,None] * scaledQ[None,...])
                    expm = matrix_exponential.matrix_exponential
                    P_a = expm(t[:, None, None, None] * scaledQ[None, ...])

                A_a = []

                for i in range(len(e_s)):
                    b = e_s[i]
                    if b < n:
                        with tf.name_scope(f"X_{b}"):
                            X_b = X[:, b, :]
                        with tf.name_scope(f"P_{b}{a}"):
                            P_ab = P_a[i, ...]
                        A_ab = tf.einsum("mcd,id -> imc", P_ab, X_b, name = f"A_{b}{a}")
                        A_a.append(A_ab)
                    else:
                        with tf.name_scope(f"P_{b}{a}"):
                            P_ab = P_a[i, ...]
                        A_ab = tf.einsum("mcd,imd -> imc", P_ab, A[b-n], name = f"A_{b}{a}")
                        A_a.append(A_ab)
                with tf.name_scope(f"A_{a}"):
                    A.append( tf.math.reduce_prod(tf.stack(A_a), axis=0) )

        P_X = tf.einsum("imc, mc -> im", A[-1], pi,
                        name = f"probability_of_data_given_model")

        # regularization loss
        self.add_loss(self.regularization_loss(Q))

        return P_X
