# hw4

## 版本

- Python 3.10.5

- Django 4.1

## 依赖

已导出为requestments.txt

可通过命令

'pip install -r requirements.txt '

来安装依赖库（主要为torch），其实就是第三次作业+Django，理论上不需要其他的库文件

## 使用

通过在项目根目录下，使用命令

'''

python manage.py makemigrations
python manage.py migrate
py manage.py runserver

''''

即可在localhost:8000 端口访问到该网站。
