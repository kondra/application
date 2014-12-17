import numpy as np

__DEBUG__ = False

def decode(c):
    return ord(c)-ord('a')

def encode(i):
    return str(unichr(ord('a')+i))

def compute_stats(filename):
    f = open(filename, 'r')

    u = np.zeros(26)
    b = np.zeros((26,26))
    n = 0

    for line in f.readlines():
        line = line.strip()
        u[decode(line[0])] += 1
        for i in xrange(1, len(line)):
            b[decode(line[i-1]),decode(line[i])] += 1
            n += 1

    if __DEBUG__:
        for i in xrange(26):
            print '{}: {}'.format(encode(i), u[i]/n)

    f.close()

    return u,b,n

def compute_loglikelihood(text, u, b, n, f):
    ll = 0
    for w in text.split('\n'):
        if len(w) == 0:
            break
        ll += np.log(u[f(decode(w[0]))]) - np.log(n)
        for i in xrange(1, len(w)):
            ll += np.log(b[f(decode(w[i-1])),f(decode(w[i]))]) - np.log(u[f(decode(w[i-1]))])
    return ll

def compute_unigram_loglikelihood(text, u, n, f):
    ll = 0
    for w in text.split(' '):
        for i in xrange(0, len(w)):
            ll += np.log(u[f(decode(w[i]))]) - np.log(n)
    return ll

def compute_ratio(encrypted_text, text, f):
    r = 0.0
    n = 0.0
    for c1,c2 in zip(encrypted_text, text):
        if decode(c2) < 26 and decode(c2) >= 0:
            r += f(decode(c1)) == decode(c2)
            n += 1
    return float(r)/n

def metropolis(text, u, b, n, max_iter=10000, display=False):
    best_f = None
    ll_sum = 0
    ll_max = -np.Inf
    f = np.arange(0, 26, dtype=np.int32)
    f = np.random.permutation(f)
    n_iter = 0
    while n_iter < max_iter:
        ll = compute_loglikelihood(text, u, b, n, lambda c: f[c])
        ll_sum += ll
        if display:
            print 'iteration: {}, LL: {}, avg LL: {}'.format(n_iter, ll, ll_sum/(n_iter+1))
        if ll > ll_max:
            ll_max = ll
            best_f = f.copy()
        i_1 = np.random.randint(26)
        i_2 = np.random.randint(26)
        f_new = f.copy()
        f_new[i_1], f_new[i_2] = f_new[i_2], f_new[i_1]
        ll_new = compute_loglikelihood(text, u, b, n, lambda c: f_new[c])
        p = min(0, ll_new - ll)
        if np.log(np.random.rand()) < p:
            f = f_new.copy()
        n_iter += 1

    return best_f, ll_max

def main(text_filename, encrypted_text_filename, true_text_filename, max_iter=5000, text_size=200, n_starts=10):
    u,b,n = compute_stats(text_filename)
    u += 0.5
    b += 0.5
    n += n*0.5

    f = open(true_text_filename, 'r')
    true_text = f.read()
    f.close()
    f = open(encrypted_text_filename, 'r')
    encrypted_text = f.read()
    f.close()

    encrypted_text = encrypted_text[:text_size]
    true_text = true_text[:text_size]

    f_best = None
    ll_best = -np.Inf
    cum_ratio = 0.0

    for i in xrange(n_starts):
        f, ll = metropolis(encrypted_text, u, b, n, max_iter=max_iter)
        ratio = compute_ratio(encrypted_text, true_text, lambda c: f[c])

        print 'log-likelihood: {}'.format(ll)
        print 'correctly discovered symbols ratio: {}'.format(ratio)

        cum_ratio += ratio

        if ll > ll_best:
            ll_best = ll
            f_best = f

    avg_ratio = cum_ratio / n_starts
    f = f_best
    ll = ll_best

    print 'average ratio of correctly discovered symbols: {}'.format(avg_ratio)

    if __DEBUG__:
        for i in xrange(26):
            print '{} -> {}'.format(encode(i), encode(f[i]))

    if __DEBUG__:
        r_true = ['c', 'p', 'v', 'l', 'd', 'o', 'x', 'a', 'e', 'q', 'z', 'm', 'y', 'n', 's', 'b', 'w', 'g', 't', 'r', 'j', 'f', 'k', 'u', 'i', 'h']
        f_true = np.zeros(26, dtype=np.int32)
        for i in xrange(26):
            f_true[decode(r_true[i])] = i
        print 'TRUE LL: {}'.format(compute_loglikelihood(encrypted_text, u, b, n, lambda c: f_true[c]))
        print 'TRUE: correctly discovered symbols ratio: {}'.format(compute_ratio(encrypted_text, true_text, lambda c: f_true[c]))


if __name__ == '__main__':
    main(
        'app_main/war_and_peace.txt',
        'app_main/oliver_twist.txt.enc',
        'app_main/oliver_twist.txt',
        10000,
        500,
        10)
