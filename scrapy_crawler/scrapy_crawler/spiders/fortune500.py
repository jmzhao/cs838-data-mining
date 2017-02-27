# -*- coding: utf-8 -*-
import scrapy
#from bs4 import BeautifulSoup


class Fortune500Spider(scrapy.Spider):
    name = "fortune500"
    allowed_domains = ["beta.fortune.com"]
    #start_urls = ['http://beta.fortune.com/fortune500/list']
    start_urls=[line.strip() for line in open('fortune500url.txt')]
    #print (start_urls)
    custom_settings = {
        'DOWNLOAD_DELAY': 0.2,
    }
    def parse(self, response):
       # title_node = response.xpath('//a[@class="question-hyperlink"]')[0]
       # print ("+++++++++++++++",title_node )
        yield {
                'intro' : ''.join(response.xpath('//div[@class="columns small-12 company-info-card-desc"]//span//p//text()').extract())
                #'intro' : BeautifulSoup(''.join(response.xpath('//div[@class="columns small-12 company-info-card-desc"]//span//p').extract_first())).get_text()
       #         'title' : title_node.xpath('text()').extract_first(),
        #        'post-text' : response.xpath('//div[@class="question"]').xpath('.//div[@class="post-text"]').extract_first(),
         #       'answer-text': response.xpath('//div[@class="answer"]').xpath('.//div[@class="post-text"]').extract_first(),
           # 'post-vote' : response.xpath('//div[@class="vote"]').xpath('.//span[@class="vote-count-post high-scored-post"]/text()').extract_first()
           #     'post-vote' : response.xpath('//span[@itemprop="upvoteCount"]').xpath('text()').extract_first(),
          #      'answer-vote' : response.xpath('//span[@itemprop="upvoteCount"]').xpath('text()').extract()[1]
        }
        pass
