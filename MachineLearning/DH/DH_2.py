  while true:
         U = U1.copy()
         delta = 0
         for s in mdp.states:
              U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            # ある地点(state)において実行し得る、各行動について:　for a in mdp.actions(s)
            # それぞれ実行し(1マスの移動(3方向)を計算し)、その結果得られる各マスとそこへの移動確率のペアをもとに:  for (p, s1) in T(s, a)
            # 移動先のマスへの移動確率を加重: p * U[s1]
            # 行動ごと計算していたのを合計し、その中で、もっとも取り得る確率が高いものを選択: max(sum[ ~ in mdp.actions(s)])
            # 現在地の報酬に、その"取り得る確率"を薄めた(*gammaした)ものを加算し、そのマスの新たな"取り得る確率 = 移動確率"とする。
            # この処理を、変化率が特定の閾値を切るまで(雑に言うとゼロに近づくまで)繰り返す。
             delta = max(delta, abs(U1[s] - U[s]))
         if delta < epsilon * (1 - gamma) / gamma:
             return U
