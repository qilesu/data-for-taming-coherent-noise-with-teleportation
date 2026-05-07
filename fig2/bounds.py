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
        epsilon = max_{t, P={I, X, Y, Z}} ([E_t]_{P,XP} / [E_t]_{P,P}, [E_t]_{ZP,XP} / [E_t]_{P,P}, [E_t]_{ZP,P} / [E_t]_{P,P})

    Parameters
    ----------
    channels : list of qutip superoperators, len = T
        The index for time step t is t-1.

    Returns
    -------
    epsilon : float
        The smallness parameter
    '''
    I, X, Y, Z = qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()
    paulis = [I, X, Y, Z]
    diagonal_part = np.array([[get_ptm_element(channel, P, P) for P in paulis] for channel in channels]) # [E_t]_{P,P}, shape = (T, 4)
    x_coherence = np.array([[get_ptm_element(channel, P, X*P) for P in paulis] for channel in channels]) # [E_t]_{P,XP}, shape = (T, 4)
    y_coherence = np.array([[get_ptm_element(channel, Z*P, X*P) for P in paulis] for channel in channels]) # [E_t]_{ZP,XP}, shape = (T, 4)
    z_coherence = np.array([[get_ptm_element(channel, Z*P, P) for P in paulis] for channel in channels]) # [E_t]_{ZP,P}, shape = (T, 4)

    smallness = np.stack([np.abs(x_coherence / diagonal_part), np.abs(y_coherence / diagonal_part), np.abs(z_coherence / diagonal_part)])
    epsilon = np.max(smallness)
    return epsilon

def get_hadamard_conjugated_channel(channels, t):
    '''Computes the Hadamard conjugated channel

    Notes
    -----
    Computes E_t^H = H^t * E_t * H^t

    Parameters
    ----------
    channels : list of qutip superoperators, len = T
        The index for time step t is t-1.
    t : int
        Time step
    
    Returns
    -------
    channel_H : qutip superoperator
        E_t^H = H^t * E_t * H^t
    '''
    H = qt.Qobj(np.array([[1, 1], [1, -1]]) / np.sqrt(2)) # define hadamard operator
    H_ch = qt.sprepost(H, H) # define hadamard channel
    channel_H = channels[t - 1]
    if t % 2 == 1:#
        channel_H = H_ch * channel_H * H_ch # channel at timestep t is index by index t-1
    return channel_H



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