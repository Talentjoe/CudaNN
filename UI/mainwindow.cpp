#include "mainwindow.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QColorDialog>
#include <QFileDialog>
#include <QPainter>
#include <QMouseEvent>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), drawing(false)
{
    // 初始化画布
    canvas = QImage(size(), QImage::Format_ARGB32_Premultiplied);
    canvas.fill(Qt::white); // 白色背景

    setupUI();
}
MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // 创建中央部件
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    // 创建画布显示区域
    pixelInfoLabel = new QLabel("Pixel Info: ", this);
    layout->addWidget(pixelInfoLabel);

    // 创建按钮
    QPushButton *clearBtn = new QPushButton("Clear", this);
    layout->addWidget(clearBtn);

    // 连接信号槽
    connect(clearBtn, &QPushButton::clicked, [this]() {
        canvas.fill(Qt::white);
        update();
    });
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
        QPainter painter(&canvas);
        painter.setPen(QPen(Qt::black, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        
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