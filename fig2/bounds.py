import numpy as np
import qutip as qt
import warnings

def get_ptm_element(channel, P1, P2):
    '''Computes E_{P1, P1} = Tr[P1.dag() * channel(P2)] / 2

    Parameters
    ----------
    channel : qutip superoperator
    P1, P2 : qutip operator

    Returns
    -------
    E_{P1, P1} = Tr[P1.dag() * channel(P2)] / 2
    '''
    return (P1.dag() * channel(P2)).tr() / 2


def get_epsilon(channels):
    '''Computes the smallness parameter of coherence
    
    Notes
    -----
    The smallness parameter is defined as
        epsilon = max_{t, P={I, X, Y, Z}} ([N_t]_{P,XP} / [N_t]_{P,P}, [N_t]_{ZP,XP} / [N_t]_{P,P}, [N_t]_{ZP,P} / [N_t]_{P,P})

    Parameters
    ----------
    channels : list of qutip superoperators

    Returns
    -------
    epsilon : float
        The smallness parameter
    '''
    I, X, Y, Z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [I, X, Y, Z]
    diagonal_part = np.array([[get_ptm_element(channel, P, P) for P in paulis] for channel in channels]) # [N_t]_{P,P}, shape = (T, 4)
    x_coherence = np.array([[get_ptm_element(channel, P, X*P) for P in paulis] for channel in channels]) # [N_t]_{P,XP}, shape = (T, 4)
    y_coherence = np.array([[get_ptm_element(channel, Z*P, X*P) for P in paulis] for channel in channels]) # [N_t]_{ZP,XP}, shape = (T, 4)
    z_coherence = np.array([[get_ptm_element(channel, Z*P, P) for P in paulis] for channel in channels]) # [N_t]_{ZP,P}, shape = (T, 4)

    smallness = np.stack([np.abs(x_coherence / diagonal_part), np.abs(y_coherence / diagonal_part), np.abs(z_coherence / diagonal_part)])
    epsilon = np.max(smallness)
    return epsilon


def H_conj(channel):
    '''For channel N, computes the Hadamard conjugated channel H N H'''
    H = qt.Qobj(np.array([[1, 1], [1, -1]]) / np.sqrt(2)) # define hadamard operator
    H_ch = qt.sprepost(H, H) # define hadamard channel
    channel_H = H_ch * channel * H_ch
    return channel_H


def get_recursion_coefficients(channels):
    '''Computes the recursion coefficient matrix A_t
    
    Parameters
    ----------
    channels : list of qutip superoperators, len = T,
        channels[t-1] is the error channel at time step t.
    
    Returns
    -------
    A : ndarray of float, shape = (4, T, 2, 2)
        for P = I,X,Y,Z ~ 0,1,2,3, and odd or even t = 1,2,...,T
        A[P, t-1, 0, 0] = a_t = [N_t^H]_{P,P} or [N_t]_{P,P}
        A[P, t-1, 1, 0] = b_t = [N_t^H]_{XP,P} or [N_t]_{ZP,P}
        A[P, t-1, 0, 1] = c_t = [N_t^H]_{P,ZP} or [N_t]_{P,XP} 
        A[P, t-1, 1, 1] = d_t = [N_t^H]_{XP,ZP} or [N_t]_{ZP,XP}
    '''
    T = len(channels)
    I, X, Y, Z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [I, X, Y, Z]

    A = np.zeros(shape=(4, len(channels), 2, 2), dtype=complex) # shape = (4, T, 2, 2)
    for t in range(1, T+1):
        if t % 2 == 1:
            channel = H_conj(channels[t-1])
            A[:, t-1, 0, 0] = np.array([get_ptm_element(channel, P, P) for P in paulis]) # shape = (4,)
            A[:, t-1, 1, 0] = np.array([get_ptm_element(channel, X*P, P) for P in paulis]) # shape = (4,)
            A[:, t-1, 0, 1] = np.array([get_ptm_element(channel, P, Z*P) for P in paulis]) # shape = (4,)
            A[:, t-1, 1, 1] = np.array([get_ptm_element(channel, X*P, Z*P) for P in paulis]) # shape = (4,)
        else:
            channel = channels[t-1]
            A[:, t-1, 0, 0] = np.array([get_ptm_element(channel, P, P) for P in paulis]) # shape = (4,)
            A[:, t-1, 1, 0] = np.array([get_ptm_element(channel, Z*P, P) for P in paulis]) # shape = (4,)
            A[:, t-1, 0, 1] = np.array([get_ptm_element(channel, P, X*P) for P in paulis]) # shape = (4,)
            A[:, t-1, 1, 1] = np.array([get_ptm_element(channel, Z*P, X*P) for P in paulis]) # shape = (4,)
    return A


def get_effective_pauli_channels(channels, order, epsilon):
    '''Computes the diagonal elements of the effective Pauli errors
    
    Parameters
    ----------
    channels : list of qutip superoperators, len = T,
        channels[t-1] is the error channel at time step t.
    order : int, >= 2
        the order of the approximation.

    Returns
    -------
    P_eff : ndarray of float, shape = (2, 4, T)
        for P = I,X,Y,Z ~ 0,1,2,3, and odd or even t = 1,2,...,T

    '''
    assert order >= 2, "order must be >= 2"
    T = len(channels)
    A = get_recursion_coefficients(channels) # shape = (4, T, 2, 2)

    z_of_epsilon = (1/epsilon - epsilon - np.sqrt((1/epsilon - epsilon)**2 - 4))/2
    error_mag = epsilon * z_of_epsilon *  ((epsilon + epsilon**2) / (1 - z_of_epsilon * epsilon)**2)**(order-1)

    P_eff = np.zeros(shape=(2, 4, T), dtype=complex) # shape = (2, 4, T), axis correspond to uppervslower-boud, Paulis, time step
    P_eff[:, :, T-1] = A[None, :, T-1, 0, 0] # shape = (2, 4), last time step is just a_T
    for t in range(1, T): # t = 1, 2,..., T-1
        # define the left and right limits of the matrix multiplication
        left = min(T, t + order - 1) # cap it off at T.
        right = t + 1

        # compute the truncation error
        if left < T: # truncation error is non-zero if left < T
            error = np.array([-1, 1]) * error_mag
        else: # there is no truncation error if left = T
            error = np.array([0, 0])

        # start with the vector v = [1, 0] and multiply by
        # v @ A_left @ A_left-1 @ ... @ A_right.
        v = np.array([[1, 0]] * 4) # shape = (4, 2), 4 corresponds to Paulis, and 2 corresponds to the axis for matrix multiplication
        for t_prime in range(left, right-1, -1): # t_prime = left, left-1, ..., right
            v = np.einsum('Pi,Pij->Pj', v, A[:, t_prime - 1])
        assert v.shape == (4, 2), f"v.shape = {v.shape}, expected (4, 2)"

        # # compute the effective Pauli channel
        P_eff[:, :, t-1] = (1 + v[:, 1] / v[:, 0] * A[:, t-1, 1, 0] / A[:, t-1, 0, 0] + error[:, None]) * A[:, t-1, 0, 0]
        assert np.allclose(np.imag(P_eff), 0)

    P_eff = np.real(P_eff) # shape = (2, 4, T)
    return P_eff

    
def get_average_channel_bounds(channels, order):
    epsilon = get_epsilon(channels)
    if epsilon >= 1/3:
        warnings.warn(f'epsilon = {epsilon} is not smaller than 1/3')
    else:
        print(f'epsilon = {epsilon}')

    T = len(channels)
    N_tot = np.zeros(shape=(2, 4, T + 1), dtype=float) # N_tot is 0 index and the ith element in the time axis is the step t=i
    N_tot[:, :, 0] = 1.0
    for t in range(1, T + 1):
        P_eff = get_effective_pauli_channels(channels[:t], order, epsilon) # compute the channels for the first t errors
        N_tot[:, :, t] = np.prod(P_eff, axis=-1) # multiply together all the channels.
    return N_tot





















































































################
# Loose bounds #
################
def get_s2t_loose(channels, t, P):
    '''Computes sign s2 = prod_{t'=1}^t sign([E_{t'}^H]_{P,P})

    Parameters
    ----------
    channels : list of qutip superoperators, len = T
        The index for time step t is t-1.
    t : int
        Time step
    P : qutip operator
        Pauli operator
    
    Returns
    -------
    s2t : 2d array of float, len = (T, 4)
        The index for time step t is t-1.
    '''
    s2t = 1
    for t_prime in range(1, t + 1):
        E_t_prime_H = get_hadamard_conjugated_channel(channels, t_prime)
        diagonal_part = np.real(get_ptm_element(E_t_prime_H, P, P))
        s2t = s2t * np.sign(diagonal_part) # s_{2,t} = s_{2,t-1} * sign(E_t^H_{P,P})
    return s2t

def get_s1t_loose(channels, t, P):
    X, Z = qt.sigmax(), qt.sigmaz()

    E_t_H = get_hadamard_conjugated_channel(channels, t)
    E_t_m1_H = get_hadamard_conjugated_channel(channels, t - 1)
    if t % 2 == 0:
        s1t = np.sign(np.real(get_ptm_element(E_t_H, P, X*P) * get_ptm_element(E_t_m1_H, X*P, P)))
    else:
        s1t = np.sign(np.real(get_ptm_element(E_t_H, P, Z*P) * get_ptm_element(E_t_m1_H, Z*P, P)))
    
    s1t *= get_s2t_loose(channels, t-2, P)
    return s1t

def get_P_t_loose(channels, t, P, epsilon):
    if t == 1:
        E_t_H = get_hadamard_conjugated_channel(channels, t)
        first_term = get_ptm_element(E_t_H, P, P)
        P_t_l = np.real(first_term)
        P_t_u = np.real(first_term)
        return P_t_l, P_t_u
    else:
        X, Z = qt.sigmax(), qt.sigmaz()
        E_t_H = get_hadamard_conjugated_channel(channels, t)
        E_t_m1_H = get_hadamard_conjugated_channel(channels, t - 1)

        first_term = np.real(get_ptm_element(E_t_H, P, P))

        if t % 2 == 0:
            second_term_numerator =  np.real(get_ptm_element(E_t_H, P, X*P) *  get_ptm_element(E_t_m1_H, X*P, P))
            second_term_denominator = np.real(get_ptm_element(E_t_m1_H, P, P))
        else:
            second_term_numerator =  np.real(get_ptm_element(E_t_H, P, Z*P) *  get_ptm_element(E_t_m1_H, Z*P, P))
            second_term_denominator = np.real(get_ptm_element(E_t_m1_H, P, P))
        
        s1t = get_s1t_loose(channels, t, P)
        s2t = get_s2t_loose(channels, t, P)
        
        second_term_l_p = second_term_numerator / second_term_denominator / (1 + 3 * s1t * epsilon**2)
        second_term_u_p = second_term_numerator / second_term_denominator / (1 - 3 * s1t * epsilon**2)

        third_term_l_p = - s2t * 3 * epsilon**3 / (1 - 3 * epsilon**2) * np.real(get_ptm_element(E_t_H, P, P))
        third_term_u_p = + s2t * 3 * epsilon**3 / (1 - 3 * epsilon**2) * np.real(get_ptm_element(E_t_H, P, P))
        
        P_t_l = first_term + second_term_l_p + third_term_l_p
        P_t_u = first_term + second_term_u_p + third_term_u_p

        return P_t_l, P_t_u

def get_loose_bounds_on_N(channels):
    T = len(channels)
    epsilon = get_epsilon(channels)
    if epsilon > 1/3:
        warnings.warn(f'epsilon = {epsilon} is larger than 1/3')
    I, X, Y, Z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [I, X, Y, Z]

    N_t = np.zeros(shape=(T + 1, 4, 2), dtype=float)
    N_t[0,:,:] = 1.0
    for t in range(1, T + 1):
        for ind, P in enumerate(paulis):
            P_t_l, P_t_u = get_P_t_loose(channels, t, P, epsilon)
            
            if P_t_l > 0:
                N_t[t,ind,0] = P_t_l * N_t[t-1,ind,0]
            else:
                N_t[t,ind,0] = P_t_l * N_t[t-1,ind,1]
            
            if P_t_u > 0:
                N_t[t,ind,1] = P_t_u * N_t[t-1,ind,1]
            else:
                N_t[t,ind,1] = P_t_u * N_t[t-1,ind,0]
            
    return N_t



################
# Tight bounds #
################
def get_r_t(channels, t, P):
    X, Z = qt.sigmax(), qt.sigmaz()
    E_t_H = get_hadamard_conjugated_channel(channels, t)
    E_t_m1_H = get_hadamard_conjugated_channel(channels, t - 1)
    if t % 2 == 0:
        return np.real(get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m1_H, P, P)) \
            + np.real(get_ptm_element(E_t_H, P, X*P) * get_ptm_element(E_t_m1_H, X*P, P))
    else:
        return np.real(get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m1_H, P, P)) \
            + np.real(get_ptm_element(E_t_H, P, Z*P) * get_ptm_element(E_t_m1_H, Z*P, P))

def get_gamma_t(channels, t, P):
    X, Z = qt.sigmax(), qt.sigmaz()
    E_t_H = get_hadamard_conjugated_channel(channels, t)
    E_t_m1_H = get_hadamard_conjugated_channel(channels, t - 1)
    if t % 2 == 0:
        return get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m1_H, P, Z*P) \
            + get_ptm_element(E_t_H, P, X*P) * get_ptm_element(E_t_m1_H, X*P, Z*P)
    else:
        return get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m1_H, P, X*P) \
            + get_ptm_element(E_t_H, P, Z*P) * get_ptm_element(E_t_m1_H, Z*P, X*P)

def get_delta_t(channels, t, P):
    X, Z = qt.sigmax(), qt.sigmaz()
    E_t_H = get_hadamard_conjugated_channel(channels, t)
    E_t_m1_H = get_hadamard_conjugated_channel(channels, t - 1)
    if t % 2 == 0:
        return get_ptm_element(E_t_H, Z*P, P) * get_ptm_element(E_t_m1_H, P, P) \
            + get_ptm_element(E_t_H, Z*P, X*P) * get_ptm_element(E_t_m1_H, X*P, P)
    else:
        return get_ptm_element(E_t_H, X*P, P) * get_ptm_element(E_t_m1_H, P, P) \
            + get_ptm_element(E_t_H, X*P, Z*P) * get_ptm_element(E_t_m1_H, Z*P, P)

def get_s2t_tight(channels, t, P):
    '''Computes sign s2 = prod_{t'=1}^t sign([E_{t'}^H]_{P,P})

    Parameters
    ----------
    channels : list of qutip superoperators, len = T
        The index for time step t is t-1.
    t : int
        Time step
    P : qutip operator
        Pauli operator
    
    Returns
    -------
    s2t : 2d array of float, len = (T, 4)
        The index for time step t is t-1.
    '''
    s2t = 1
    for t_prime in range(1, (t-2) + 1):
        E_t_prime_H = get_hadamard_conjugated_channel(channels, t_prime)
        diagonal_part = np.real(get_ptm_element(E_t_prime_H, P, P))
        s2t = s2t * np.sign(diagonal_part) # *= sign(E_t'^H_{P,P}), up to t' = t-2
    
    E_t_m2_prime_H = get_hadamard_conjugated_channel(channels, t-2)
    diagonal_part = np.real(get_ptm_element(E_t_m2_prime_H, P, P))
    s2t = s2t * np.sign(diagonal_part) # *= sign(E_(t-2)^H_{P,P})
    
    E_t_prime_H = get_hadamard_conjugated_channel(channels, t)
    diagonal_part = np.real(get_ptm_element(E_t_prime_H, P, P))
    s2t = s2t * np.sign(diagonal_part) # *= sign(E_t^H_{P,P})
    return s2t

def get_s1t_tight(channels, t, P):
    s1t = 1
    for t_prime in range(1, (t-4) + 1):
        E_t_prime_H = get_hadamard_conjugated_channel(channels, t_prime)
        diagonal_part = np.real(get_ptm_element(E_t_prime_H, P, P))
        s1t = s1t * np.sign(diagonal_part) # *= sign(E_t'^H_{P,P}), up to t' = t-4
    
    s1t *= np.sign(np.real(get_gamma_t(channels, t, P) * get_delta_t(channels, t-2, P)))
    return s1t

def get_P_t_tight(channels, t, P, epsilon):
    X, Z = qt.sigmax(), qt.sigmaz()
    if t == 1:
        E_t_H = get_hadamard_conjugated_channel(channels, t)
        first_term = np.real(get_ptm_element(E_t_H, P, P))
        P_t_l = first_term
        P_t_u = first_term
        return P_t_l, P_t_u
    elif t == 2:
        first_term = get_r_t(channels, t=2, P=P)
        P_t_l = first_term
        P_t_u = first_term
        return P_t_l, P_t_u
    elif t == 3:
        E_1_H = get_hadamard_conjugated_channel(channels, 1)
        first_term = get_r_t(channels, t=3, P=P)
        second_term_numerator = np.real(get_gamma_t(channels, t=3, P=P) * get_ptm_element(E_1_H, X*P, P))
        second_term_denominator = np.real(get_ptm_element(E_1_H, P, P))
        P_t_l = first_term + second_term_numerator / second_term_denominator
        P_t_u = first_term + second_term_numerator / second_term_denominator
        return P_t_l, P_t_u
    else:
        E_t_H = get_hadamard_conjugated_channel(channels, t)
        E_t_m2_H = get_hadamard_conjugated_channel(channels, t - 2)
        E_t_m3_H = get_hadamard_conjugated_channel(channels, t - 3)

        first_term = get_r_t(channels, t, P)

        second_term_numerator =  np.real(get_gamma_t(channels, t, P) * get_delta_t(channels, t-2, P))
        second_term_denominator = np.real(get_ptm_element(E_t_m2_H, P, P) * get_ptm_element(E_t_m3_H, P, P))
        
        s1t = get_s1t_tight(channels, t, P)
        s2t = get_s2t_tight(channels, t, P)
        
        second_term_l_p = second_term_numerator / second_term_denominator / (1 + 5 * s1t * epsilon**2)
        second_term_u_p = second_term_numerator / second_term_denominator / (1 - 5 * s1t * epsilon**2)

        third_term_l_p = - s2t * 18 * epsilon**4 * np.real(get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m2_H, P, P))
        third_term_u_p = + s2t * 18 * epsilon**4 * np.real(get_ptm_element(E_t_H, P, P) * get_ptm_element(E_t_m2_H, P, P))
        
        P_t_l = first_term + second_term_l_p + third_term_l_p
        P_t_u = first_term + second_term_u_p + third_term_u_p

        return P_t_l, P_t_u

def get_tight_bounds_on_N(channels):
    T = len(channels)
    epsilon = get_epsilon(channels)
    if epsilon > 1/3:
        warnings.warn(f'epsilon = {epsilon} is larger than 1/3')
    I, X, Y, Z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [I, X, Y, Z]

    N_t = np.zeros(shape=(T + 1, 4, 2), dtype=float)
    N_t[0,:,:] = 1.0
    for t in range(1, T + 1):
        for ind, P in enumerate(paulis):
            P_t_l, P_t_u = get_P_t_tight(channels, t, P, epsilon)
            if t == 1:
                N_t[1,ind,0] = P_t_l * N_t[0,ind,0]
                N_t[1,ind,1] = P_t_l * N_t[0,ind,1]
            else:
                if P_t_l > 0:
                    N_t[t,ind,0] = P_t_l * N_t[t-2,ind,0]
                else:
                    N_t[t,ind,0] = P_t_l * N_t[t-2,ind,1]
                
                if P_t_u > 0:
                    N_t[t,ind,1] = P_t_u * N_t[t-2,ind,1]
                else:
                    N_t[t,ind,1] = P_t_u * N_t[t-2,ind,0]
    return N_t