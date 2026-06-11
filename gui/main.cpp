#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    MainWindow window;
    window.setWindowTitle("Kuba GUI");
    window.resize(1100, 750);
    window.show();
    
    return app.exec();
}