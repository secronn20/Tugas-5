from youtube_comment_downloader import YoutubeCommentDownloader
import csv

downloader = YoutubeCommentDownloader()

url = "https://youtu.be/0YLSPyGA4h0"

comments = downloader.get_comments_from_url(url)

with open("komentar_youtube_agaklaen_1.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["platform", "komentar"])

    count = 0
    for comment in comments:
        writer.writerow(["YouTube", comment["text"]])
        count += 1

print("Jumlah komentar berhasil diambil:", count)