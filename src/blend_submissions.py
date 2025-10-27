import pandas as pd

def main():
    # Load submissions
    q = pd.read_csv('submission_deberta.csv')
    l = pd.read_csv('submission_qwen3.csv')
    m = pd.read_csv('submission_distilroberta.csv')
    w = pd.read_csv('submission_qwen14b.csv')
    a = pd.read_csv('submission_debertaauc.csv')
    al = pd.read_csv('submission_albert.csv')
    sq = pd.read_csv('submission_squeezebert.csv')

    # Rank and normalize
    rq = q['rule_violation'].rank(method='average') / (len(q) + 1)
    rl = l['rule_violation'].rank(method='average') / (len(l) + 1)
    rm = m['rule_violation'].rank(method='average') / (len(m) + 1)
    rw = w['rule_violation'].rank(method='average') / (len(w) + 1)
    ra = a['rule_violation'].rank(method='average') / (len(a) + 1)
    ral = al['rule_violation'].rank(method='average') / (len(al) + 1)
    rsq = sq['rule_violation'].rank(method='average') / (len(sq) + 1)

    # Blend with weights
    blend = (0.25 * rq + 0.15 * rl + 0.15 * rm + 0.15 * rw + 
             0.15 * ra + 0.10 * ral + 0.05 * rsq)
    
    q['rule_violation'] = blend
    q.to_csv('submission.csv', index=False)
    print("Blended submission saved to submission.csv")

if __name__ == "__main__":
    main()