# this is not conceptually correct... just out of curiosity for now...
# prob over time Vs prob over "space" i.e. being |0> or |1> ...
p = plot(
    hatpsisquarednorm(t, 0.01)*100, prob_1_unitary(t),
    (t, -2, 20*pi),
    show=False
)

p[0].line_color = 'red'
p[1].line_color = 'green'
p.show()

