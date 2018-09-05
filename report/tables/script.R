squad <- read.csv("Desktop/result_squad.csv")
plot(squad$epoch, squad$f1, xlab="Epoch",ylab="f1",
     main="SQuAD Training Curve", col="red")
lines(squad$epoch, squad$f1, col="red")
legend("bottomright", col="red", lty=1, legend="SQuAD")


multirc <- read.csv("Desktop/result_multirc.csv")
plot(multirc$epoch, multirc$sent, xlab="Epoch", ylab="f1m",
     main="MultiRC Training Curve", col=c("red"), ylim=c(0.4,0.8))
points(multirc$epoch, multirc$ans, col=c("blue"))
lines(multirc$epoch, multirc$sent, col="red")
lines(multirc$epoch, multirc$ans, col="blue")
legend("bottomright", col=c("red","blue"), lty=1,
       legend=c("Sentences", "Answers"))
