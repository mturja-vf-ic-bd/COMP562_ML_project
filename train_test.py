import torch.nn.functional as F
from settings import args
import time


def train(train_loader, model, optimizer, scheduler, epoch):
    scheduler.step()
    model.train()
    for name, param in model.named_parameters():
        print(name, type(param.data), param.size())
    start = time.time()
    train_loss, n_samples = 0, 0
    for batch_idx, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, data[4])
        loss.backward()
        optimizer.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                time_iter / (batch_idx + 1)))


def test(test_loader, model, epoch):
    model.eval()
    start = time.time()
    test_loss, correct, n_samples = 0, 0, 0
    for batch_idx, data in enumerate(test_loader):
        for i in range(len(data)):
            data[i] = data[i].to(args.device)
        # if args.use_cont_node_attr:
        #     data[0] = norm_features(data[0])
        output = model(data)
        loss = F.cross_entropy(output, data[4], reduction='sum')
        test_loss += loss.item()
        n_samples += len(output)
        pred = output.detach().cpu().max(1, keepdim=True)[1]

        correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

    acc = 100. * correct / n_samples
    print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \tsec/iter: {:.4f}\n'.format(
        epoch + 1,
        test_loss / n_samples,
        correct,
        n_samples,
        acc, (time.time() - start) / len(test_loader)))
    return acc
