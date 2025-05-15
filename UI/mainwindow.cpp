#include "mainwindow.h"

#include <iostream>
#include <QVBoxLayout>
#include <QPushButton>
#include <QColorDialog>
#include <QFileDialog>
#include <QPainter>
#include <QMouseEvent>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), drawing(false)
{
    nn = nullptr;
    canvas = QImage(QSize(280,280), QImage::Format_ARGB32_Premultiplied);
    canvas.fill(Qt::white);

    setupUI();
}
MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    layout->setContentsMargins(0, 280, 0, 0);

    pixelInfoLabel = new QLabel("Pixel Info: ", this);
    pixelInfoLabel->setStyleSheet("font-size: 16px; color: red;");
    layout->addWidget(pixelInfoLabel);

    QPushButton *openButton = new QPushButton("Open", this);
    layout->addWidget(openButton);

    QPushButton *guessButton = new QPushButton("Guess", this);
    layout->addWidget(guessButton);

    guessInfoLabel = new QLabel("Guess: ", this);
    layout->addWidget(guessInfoLabel);

    QPushButton *clearBtn = new QPushButton("Clear", this);
    layout->addWidget(clearBtn);

    connect(clearBtn, &QPushButton::clicked, [this]() {
        canvas.fill(Qt::white);
        update();
    });

    connect(openButton, &QPushButton::clicked, this, &MainWindow::openModule);

    connect(guessButton, &QPushButton::clicked, this, &MainWindow::guessModule);
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        lastPoint = event->pos();
        drawing = true;
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (drawing) {
        if (event -> pos().x() < 0 || event -> pos().y() < 0 || event -> pos().x() >= canvas.width() || event -> pos().y() >= canvas.height()) {
            return;
        }
        QPainter painter(&canvas);
        painter.setPen(QPen(Qt::black, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        
        // 绘制线条
        painter.drawLine(lastPoint, event->pos());
        
        // 更新最后位置
        lastPoint = event->pos();
        
        // 触发重绘
        update();

        // 获取当前像素颜色
        QColor color = canvas.pixelColor(event->pos());
        pixelInfoLabel->setText(QString("Pixel Info: (%1, %2) - R:%3 G:%4 B:%5")
                                .arg(event->x()).arg(event->y())
                                .arg(color.red()).arg(color.green()).arg(color.blue()));
    }
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        drawing = false;
        
        // 可选：保存当前绘图状态
        // canvas.save("drawing.png");
    }
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.drawImage(QPoint(0, 0), canvas);
}

void MainWindow::openModule()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Module", "", "All Files (*)");
    if (!fileName.isEmpty()) {
        nn = new NN::NNCore(fileName.toStdString(), 0.01);
    }
}

void MainWindow::guessModule() {
    if (nn == nullptr) {
        guessInfoLabel -> setText("Please load a model first.");
        return;
    }
    std::vector<float> imageList = getVector();
    nn ->forward(imageList);
    int result = nn -> choice();
    guessInfoLabel -> setText(QString("Guess: %1").arg(result));
}

std::vector<float> MainWindow::getVector() {
    std::vector<float> imageList(28*28);
    for (int i = 0; i < 280; i+=10) {
        for (int j = 0; j < 280; j+=10) {
            int temp = 0;
            for (int k = 0; k < 10; k++) {
                for (int l = 0; l < 10; l++) {
                    temp += canvas.pixelColor(k+i, l+j).red();
                }
            }
            imageList[i/10  + j/10* 28] = 1 - (static_cast<float>(temp) / 255 / 100);
        }
    }
    return imageList;
}
