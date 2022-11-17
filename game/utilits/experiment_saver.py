def save(p, k, value, time):
    file = open("data_exp/val" + str('{0:1.1f}'.format(p)) + "_" + str('{0:1.1f}'.format(k)) + ".txt", "w")

    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            a = value[i][j]
            file.write('{0:10.4f}'.format(a))
        file.write('\n')

    file.write('{0:10.4f}'.format(time))
    file.write('\n')
    file.close()
