import numpy as np
import pylab as pl

import sys

__DEBUG__ = False

def decode(c):
    return ord(c)-ord('a')

def encode(i):
    return str(unichr(ord('a')+i))

def compute_stats(filename, text_size=-1):
    f = open(filename, 'r')

    u = np.zeros(26)
    b = np.zeros((26,26))
    n = 0

    cnt = 0

    for line in f.readlines():
        cnt += 1
        if text_size > 0 and cnt > text_size:
            break
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

def compute_penalized_unigram_loglikelihood(text, u, n, f, d):
# try to use different scheme of penalizing!
    ll = 0
    for w in text.split('\n'):
        w_decoded = [encode(f(decode(c))) for c in w]
#        if w_decoded == w and w_decoded in d:
#            ll += 0.01 * np.log(d[w])
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

def metropolis(score_function, max_iter=10000, display=False, greedy=False):
    best_f = None
    ll_sum = 0
    ll_max = -np.Inf
    f = np.arange(0, 26, dtype=np.int32)
    f = np.random.permutation(f)
    n_iter = 0
    ll_history = []
    while n_iter < max_iter:
        ll = score_function(lambda c: f[c])
        ll_history.append(ll)
        if display:
            print 'iteration: {}, LL: {}, avg LL: {}'.format(n_iter, ll, np.mean(ll_history[-100:]))
        if ll > ll_max:
            ll_max = ll
            best_f = f.copy()
        i_1 = np.random.randint(26)
        i_2 = np.random.randint(26)
        f_new = f.copy()
        f_new[i_1], f_new[i_2] = f_new[i_2], f_new[i_1]
        ll_new = score_function(lambda c: f_new[c])
        if greedy and ll_new > ll:
            f = f_new.copy()
        elif not greedy:
            p = min(0, ll_new - ll)
            if np.log(np.random.rand()) < p:
                f = f_new.copy()
        n_iter += 1

    if __DEBUG__:
        pl.plot(ll_history)
        pl.show()

    return best_f, ll_max

def read_file(filename):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    return text

def collect_dictionary(text_filename, K):
    d = {}
    f = open(text_filename, 'r')
    for word in f.readlines():
        word = word.strip()
        if word not in d:
            d[word] = 0
        d[word] += 1
    sorted_d = sorted(d.items(), key=lambda x: -x[1])
    f.close()
    return dict(sorted_d[0:K])

def compute_loglikelihood_for_EM(f, u_all, b_all, n_all, encrypted_texts, z):
    L,K = z.shape
    ll = 0
    for i in xrange(L):
        for k in xrange(K):
            ll += z[i,k] * compute_loglikelihood(encrypted_texts[i], u_all[k], b_all[k], n_all[k], f)
    return ll

def main(text_filename, encrypted_text_filename, true_text_filename, max_iter=5000, text_size=-1, enc_text_size=200, n_starts=10, greedy=False, display=False):
# base task; metropolis with bigrams
    u,b,n = compute_stats(text_filename, text_size)
    u += 0.5
    b += 0.5
    n += n*0.5

    true_text = read_file(true_text_filename)
    encrypted_text = read_file(encrypted_text_filename)

    true_text = true_text[:enc_text_size]
    encrypted_text = encrypted_text[:enc_text_size]

    f_best = None
    ll_best = -np.Inf
    cum_ratio = 0.0
    best_ratio =-np.Inf 

    for i in xrange(n_starts):
        f, ll = metropolis(lambda f: compute_loglikelihood(encrypted_text, u, b, n, f), max_iter=max_iter, greedy=greedy, display=display)
        ratio = compute_ratio(encrypted_text, true_text, lambda c: f[c])

        print 'log-likelihood: {}'.format(ll)
        print 'ratio of correctly discovered symbols: {}'.format(ratio)

        cum_ratio += ratio

        if ll > ll_best:
            ll_best = ll
            f_best = f
            best_ratio = ratio

    avg_ratio = cum_ratio / n_starts
    f = f_best
    ll = ll_best

    print 'Best LL: {}, best ratio: {}'.format(ll_best, best_ratio)
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

def main2(text_filename, encrypted_text_filename, true_text_filename, max_iter=5000, text_size=-1, enc_text_size=200, n_starts=10, K=100, greedy=False, display=False):
# individual task E
    u,b,n = compute_stats(text_filename, text_size)
    u += 0.5
    b += 0.5
    n += n*0.5

    d = collect_dictionary(text_filename, K, text_size)
    true_text = read_file(true_text_filename)
    encrypted_text = read_file(encrypted_text_filename)

    true_text = true_text[:enc_text_size]
    encrypted_text = encrypted_text[:enc_text_size]

    f_best = None
    ll_best = -np.Inf
    ratio_best = 0.0
    cum_ratio = 0.0

    for i in xrange(n_starts):
        f, ll = metropolis(lambda f: compute_penalized_unigram_loglikelihood(encrypted_text, u, n, f, d), max_iter=max_iter, greedy=greedy, display=display)
        ratio = compute_ratio(encrypted_text, true_text, lambda c: f[c])

        print 'log-likelihood: {}'.format(ll)
        print 'correctly discovered symbols ratio: {}'.format(ratio)

        cum_ratio += ratio

        if ll > ll_best:
            ll_best = ll
            f_best = f
            ratio_best = ratio

    avg_ratio = cum_ratio / n_starts

    print 'Best LL: {}, best ratio: {}'.format(ll_best, ratio_best)
    print 'average ratio of correctly discovered symbols: {}'.format(avg_ratio)

def main3(text_filenames, encrypted_text_filenames, true_text_filenames, max_iter=5000, enc_text_size=200, text_size=-1, n_starts=10, display=False, em_max_iter=100):
# bonus task B
    u_all = []
    b_all = []
    n_all = []
    for filename in text_filenames:
        u,b,n = compute_stats(filename)
        u += 0.5
        b += 0.5
        n += n*0.5
        u_all.append(u)
        b_all.append(b)
        n_all.append(n)

    true_texts = []
    encrypted_texts = []
    for fn1, fn2 in zip(true_text_filenames, encrypted_text_filenames):
        true_texts.append(read_file(fn1)[:enc_text_size])
        encrypted_texts.append(read_file(fn2)[:enc_text_size])

    f_best = None
    ll_best = -np.Inf
    best_ratio =-np.Inf 
    f = np.arange(0, 26, dtype=np.int32)
    f = np.random.permutation(f)
    L = len(encrypted_texts)
    z = np.zeros((L, 4))

    if __DEBUG__:
        r_true = ['m', 'n', 'k', 'y', 'x', 'f', 'g', 'j', 'r', 'q', 'p', 'v', 'l', 't', 'i', 'w', 's', 'e', 'o', 'a', 'h', 'b', 'c', 'z', 'u', 'd']
        f_true = np.zeros(26, dtype=np.int32)
        for i in xrange(26):
            f_true[decode(r_true[i])] = i
        for l in xrange(L):
            for k in xrange(4):
                z[l,k] = compute_loglikelihood(encrypted_texts[l], u_all[k], b_all[k], n_all[k], lambda c: f_true[c])

        z = np.exp(z - np.max(z, axis=1, keepdims=True))
        z /= np.sum(z, axis=1, keepdims=True)

        print z

    # EM algorithm
    for n_iter in xrange(em_max_iter):
        # E-step:
        for l in xrange(L):
            for k in xrange(4):
                z[l,k] = compute_loglikelihood(encrypted_texts[l], u_all[k], b_all[k], n_all[k], lambda c: f[c])
        z = np.exp(z - np.max(z, axis=1, keepdims=True))
        z /= np.sum(z, axis=1, keepdims=True)
        print 'E-step. Latent variables distribution:'
        print z
        # M-step:
        print 'M step'
        ll_best_ = -np.Inf
        f_best_ = None
        for i in xrange(n_starts):
            f, ll = metropolis(lambda f: compute_loglikelihood_for_EM(f, u_all, b_all, n_all, encrypted_texts, z), max_iter=max_iter, display=display)
            print 'start {}: LL = {}'.format(i, ll)
            if ll > ll_best_:
                ll_best_ = ll
                f_best_ = f.copy()
        ll = ll_best_
        f = f_best_.copy()
        ratio = 0
        # all texts are of the same length
        for l in xrange(L):
            ratio += compute_ratio(encrypted_texts[k], true_texts[k], lambda c: f[c])
        ratio /= 4

        print 'log-likelihood: {}'.format(ll)
        print 'ratio of correctly discovered symbols: {}'.format(ratio)

        if ll > ll_best:
            ll_best = ll
            f_best = f
            best_ratio = ratio

    print 'Best LL: {}, best ratio: {}'.format(ll_best, best_ratio)

if __name__ == '__main__':
    if sys.argv[1] == '1':
        # main task
        main(
            'app_main/war_and_peace.txt',
            'app_main/oliver_twist.txt.enc',
            'app_main/oliver_twist.txt',
            max_iter=5000,
            text_size=200000,
            enc_text_size=8000,
            n_starts=5,
            greedy=False,
            display=False)

    if sys.argv[1] == '2':
        # individual task E
        main2(
            'app_main/war_and_peace.txt',
            'app_main/oliver_twist.txt.enc',
            'app_main/oliver_twist.txt',
            max_iter=10000,
            enc_text_size=1000,
            n_starts=10,
            display=False)

    if sys.argv[1] == '3':
        # bonus task B
        main3(
            ['app_main/war_and_peace.txt','bonus_b/de/war_and_piece.txt','bonus_b/fr/war_and_peace.txt','bonus_b/pg/text.txt'],
            ['app_main/oliver_twist.txt.enc1','bonus_b/de/enc.txt','bonus_b/fr/enc.txt','bonus_b/pg/enc.txt'],
            ['app_main/oliver_twist.txt','bonus_b/de/02.txt','bonus_b/fr/oliver_twist.txt','bonus_b/pg/pg20103.txt.done'],
            max_iter=2000,
            text_size=200000,
            enc_text_size=2000,
            n_starts=5,
            display=False,
            em_max_iter=50)
