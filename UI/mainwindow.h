#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <NNCore.cuh>
#include <QMainWindow>
#include <QImage>
#include <QPoint>
#include <QLabel>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    QImage canvas;          // 画布图像
    QPoint lastPoint;       // 上一个鼠标位置
    bool drawing;           // 绘图状态
    QLabel *pixelInfoLabel; // 像素信息显示标签
    QLabel *guessInfoLabel; // 像素信息显示标签

    NN::NNCore *nn; // 神经网络对象

    void setupUI();
    void openModule();
    void guessModule();
    std::vector<float> getVector();
};

#endif // MAINWINDOW_H